"""Run the analysis pipeline in order.

    python run_all.py              # steps 01-13
    python run_all.py --from 2     # start from step 02 (e.g. when work/analysis_dataset.pkl already exists)
    python run_all.py --only 12    # one step

Paths can be changed with the environment variables BCP_DATA, BCP_WORK and BCP_RESULTS (see pipeline/common.py).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / 'pipeline'
STEPS = sorted(PIPELINE.glob('[0-9][0-9]_*.py'))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--from', dest='first', type=int, default=1)
parser.add_argument('--to', dest='last', type=int, default=len(STEPS))
parser.add_argument('--only', type=int)
args = parser.parse_args()
first, last = (args.only, args.only) if args.only else (args.first, args.last)

for step in STEPS:
    if not first <= int(step.name[:2]) <= last:
        continue
    print(f'\n=== {step.name}', flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, step.name], cwd=PIPELINE, check=True)
    print(f'=== {step.name} finished in {time.time() - t0:.0f} s', flush=True)
