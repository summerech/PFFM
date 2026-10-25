import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge CSV files while preserving all columns.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for name in args.inputs:
        with Path(name).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    fields = sorted({key for row in rows for key in row})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(rows)} rows into {output}")


if __name__ == "__main__":
    main()
