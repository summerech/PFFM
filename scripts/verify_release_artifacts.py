"""Validate released main-run archives against the reported three-seed result."""

import csv
import hashlib
import io
import json
import statistics
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "artifacts" / "pffm_100pocket"
EXPECTED_FILES = {
    "samples.json",
    "assembled/assembled.sdf",
    "assembled/summary.csv",
    "manifest.csv",
    "docked.csv",
    "metrics.json",
    "metrics.csv",
}


def check_hashes():
    for line in (ROOT / "artifacts" / "SHA256SUMS.txt").read_text().splitlines():
        expected, relative_path = line.split(maxsplit=1)
        path = ROOT / relative_path.strip()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {relative_path}")


def main():
    check_hashes()
    values = []
    for seed in (2021, 2022, 2023):
        with ZipFile(ARCHIVES / f"seed_{seed}_raw_and_scores.zip") as archive:
            if set(archive.namelist()) != EXPECTED_FILES:
                raise ValueError(f"Unexpected archive contents for seed {seed}")
            metrics = json.load(archive.open("metrics.json"))
            reader = io.TextIOWrapper(archive.open("docked.csv"), encoding="utf-8")
            rows = list(csv.DictReader(reader))
            keys = {(int(row["pocket_index"]), int(row["sample_index"])) for row in rows}
            expected_keys = {(pocket, sample) for pocket in range(100) for sample in range(100)}
            if len(rows) != 10000 or keys != expected_keys or metrics["requested"] != 10000:
                raise ValueError(f"Incomplete or duplicated sample records for seed {seed}")
            values.append(metrics["vina_macro"])
            print(f"seed {seed}: {len(rows)} records, {metrics['scored']} scored, Vina Dock {values[-1]:.6f}")

    reported = json.loads((ROOT / "paper_results" / "esm2_full100_three_seed_summary.json").read_text())
    expected_mean = reported["metrics"]["vina_macro"]["esm2_mean"]
    actual_mean = statistics.mean(values)
    if abs(actual_mean - expected_mean) > 1e-9:
        raise ValueError("Three-seed mean differs from the paper summary")
    print(f"Verified three-seed mean: {actual_mean:.6f} kcal/mol")


if __name__ == "__main__":
    main()
