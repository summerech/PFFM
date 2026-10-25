import argparse
from collections import Counter
import json
import lmdb
import pickle
import torch


def main():
    parser = argparse.ArgumentParser(description="Build a fragment token vocabulary.")
    parser.add_argument("cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--split-file")
    parser.add_argument("--split-name", default="train")
    args = parser.parse_args()

    env = lmdb.open(
        args.cache,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    frequency = Counter()
    seen_ids = set()
    allowed_ids = None
    if args.split_file:
        allowed_ids = set(torch.load(args.split_file)[args.split_name])
    with env.begin() as transaction:
        for key, value in transaction.cursor():
            if not key.startswith(b"id:"):
                continue
            dataset_id = int(key[3:])
            if allowed_ids is not None and dataset_id not in allowed_ids:
                continue
            if dataset_id in seen_ids:
                continue
            seen_ids.add(dataset_id)
            frequency.update(pickle.loads(value)["fragment_keys"])
    env.close()

    tokens = [
        token
        for token, count in sorted(
            frequency.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= args.min_frequency
    ]
    if args.max_size is not None:
        tokens = tokens[: max(0, args.max_size - 1)]
    token_to_id = {"<unk>": 0}
    token_to_id.update({token: index + 1 for index, token in enumerate(tokens)})
    output = {
        "token_to_id": token_to_id,
        "frequency": {token: frequency[token] for token in tokens},
        "num_cached_molecules": len(seen_ids),
        "vocab_size": len(token_to_id),
        "unknown_id": 0,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({key: value for key, value in output.items() if key != "token_to_id" and key != "frequency"}, indent=2))


if __name__ == "__main__":
    main()
