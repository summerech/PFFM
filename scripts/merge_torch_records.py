import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Merge torch record datasets.")
    parser.add_argument("input", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = []
    sources = []
    for name in args.input:
        payload = torch.load(name, map_location="cpu")
        records.extend(payload["records"])
        sources.append({"path": str(name), "records": len(payload["records"])})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "sources": sources}, output)
    print({"records": len(records), "sources": sources, "output": str(output)})


if __name__ == "__main__":
    main()
