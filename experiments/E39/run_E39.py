"""
E39: posterior predictive check of the null simulators (20 simulated EL-wide data sets each).
See src/el_wide_nulls.py. Requires RESULTS_ROOT/E30/E30_fit.json.

Outputs (RESULTS_ROOT/E39): E39_ppc.json
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402
import el_wide_nulls  # noqa: E402

if __name__ == "__main__":
    out_dir = config.results_dir("E39")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(out_dir / "E39_run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    el_wide_nulls.main(mode="ppc", out_dir=out_dir, n_null_reps=200, n_workers=config.n_workers(2))
