import torch
import torch.nn.functional as F
from mock_d1.model_mock import MockD1ForCausalLM
from data.tokeniser import TokenizerManager
from .cache import MockD1StateCache

class MockD1Generator:
    def __init__(self, model: MockD1ForCausalLM, tokenizer: TokenizerManager, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        cache = MockD1StateCache(self.model.config, batch_size=1, device=self.device)

        # Prefill prompt tokens
        for i in range(input_ids.shape[1] - 1):
            curr_token = input_ids[:, i:i+1]
            _, _, new_states = self.model(curr_token, past_states=cache.states)
            cache.update(new_states)

        last_token = input_ids[:, -1:]
        generated = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            logits, _, new_states = self.model(last_token, past_states=cache.states)
            cache.update(new_states)

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            last_token = next_token

            if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated)