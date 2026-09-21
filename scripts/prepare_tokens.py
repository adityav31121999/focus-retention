"""Tokenize once on CPU, then reuse the packed file on GPU and TPU."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Hugging Face dataset name")
    source.add_argument("--text-file", help="UTF-8 text file, one document per line")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    if args.dataset:
        from datasets import load_dataset
        rows = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
        texts = (row.get(args.text_column, "") for row in rows)
    else:
        texts = open(args.text_file, encoding="utf-8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count, batch = 0, []
    # Exclusive creation prevents an accidental overwrite of an existing corpus.
    with output.open("xb") as f:
        def emit(documents):
            nonlocal count
            encoded = tokenizer(documents, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                values = (ids + [tokenizer.eos_token_id])[:args.max_tokens - count]
                np.asarray(values, dtype="<u4").tofile(f)
                count += len(values)
                if count >= args.max_tokens:
                    break
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            batch.append(text)
            if len(batch) == 256:
                emit(batch)
                batch.clear()
                if count >= args.max_tokens:
                    break
        if batch and count < args.max_tokens:
            emit(batch)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(dict(
        dtype="<u4", num_tokens=count, vocab_size=len(tokenizer), tokenizer=args.tokenizer,
        source=args.dataset or args.text_file)), encoding="utf-8")
    print(f"Wrote {count:,} tokens to {output}")


if __name__ == "__main__":
    main()
