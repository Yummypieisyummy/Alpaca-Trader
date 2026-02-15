"""
Run short-interval research with NVDA 1-min data (batch launcher)

Usage:
    python run_short_research.py
    python run_short_research.py --full    # Run with full parameter grid (slower)
"""

import subprocess
import sys
from pathlib import Path

base_path = Path(__file__).parent

csv_file = base_path.parent / "data" / "nvda_1min_2024-2025_cleaned.csv"

if not csv_file.exists():
    print(f"Error: {csv_file} not found.")
    print("First, run: python nvda_short_interval_scraper.py")
    sys.exit(1)

args = [
    sys.executable,
    str(base_path / "short_interval_research.py"),
    "--csv", str(csv_file),
    "--symbol", "NVDA"
]

if "--full" in sys.argv:
    print("Running FULL parameter grid (may take several minutes)...")
else:
    args.append("--quick")
    print("Running QUICK parameter grid...")

print(f"CSV: {csv_file}")
print(f"Args: {' '.join(args)}\n")

result = subprocess.run(args)
sys.exit(result.returncode)
