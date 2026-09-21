"""Exercise FSDP sharding/checkpoint collectives on two CPU/Gloo processes."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _fsdp_worker(rank, rendezvous, output):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from mock_d1 import MockD1Config, MockD1ForCausalLM
    from mock_d1.block import MockD1Block
    from engine.optimiser import Adafactor
    from scripts.training_common import restore, save

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        config = MockD1Config(vocab_size=32, hidden_dim=8, kqv_dim=8, intermediate_dim=16,
                              num_blocks=1, num_layers=4, focus_heads=2, retention_latent_dim=3,
                              lora_deembed_rank=2, chunk_size=3)
        base = MockD1ForCausalLM(config)
        base.gradient_checkpointing_enable()
        model = FSDP(base, device_id=torch.device("cpu"), backward_prefetch=None,
                     auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={MockD1Block}))
        optimizer = Adafactor(model.parameters(), lr=1e-3, scale_parameter=False)
        args = SimpleNamespace(output_dir=output, keep_last=2, resume=None)
        signature = dict(world_size=2, strategy="fsdp")

        def update(offset):
            ids = (torch.arange(8).view(1, 8) + rank * 8 + offset) % 32
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(ids, labels=ids, return_logits=False, loss_chunk_size=3)
            loss.backward()
            model.clip_grad_norm_(1.0)
            optimizer.step()
            return loss.detach()

        update(0)
        save(args, model, optimizer, None, signature, rank, "fsdp", 1, 1, dist.barrier)
        expected_loss = update(1)
        expected_params = [p.detach().clone() for p in model.parameters()]
        args.resume = str(Path(output) / "step-00000001")
        step, consumed = restore(args, model, optimizer, None, signature, rank, "fsdp")
        assert (step, consumed) == (1, 1)
        actual_loss = update(1)
        torch.testing.assert_close(actual_loss, expected_loss)
        for actual, expected in zip(model.parameters(), expected_params):
            torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")
def test_fsdp_two_rank_training_and_resume(tmp_path):
    mp.spawn(_fsdp_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "checkpoints")),
             nprocs=2, join=True)
