from sashimi.helicon_stack import stack_from_to
from pathlib import Path

if __name__ == "__main__":
    indir = Path(input("original dir"))
    outdir = Path(input("output dir"))

    for scan in indir.iterdir():
        if "scan" not in scan.name:
            continue
        stack_from_to(scan, outdir.joinpath(scan.stem))
