import compileall
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    ok = all(
        compileall.compile_dir(root / folder, quiet=1, force=True)
        for folder in ("models", "datasets", "utils", "scripts")
    )
    if not ok:
        raise SystemExit("Python syntax check failed")
    sys.path.insert(0, str(root))
    from models.fragment_flow import FragmentPocketFlow  # noqa: F401
    from models.topology_preserving_refiner import TopologyPreservingAtomicRefiner  # noqa: F401
    from utils.fragment_assembly import assemble_fragment_graph  # noqa: F401
    print("PFFM paper-code imports passed")


if __name__ == "__main__":
    main()
