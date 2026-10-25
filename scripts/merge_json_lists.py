import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge JSON list files in argument order.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = []
    for name in args.inputs:
        payload = json.loads(Path(name).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"expected a JSON list: {name}")
        merged.extend(payload)
    Path(args.output).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"merged {len(merged)} records into {args.output}")


if __name__ == "__main__":
    main()
