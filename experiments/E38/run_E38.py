"""
E38: the primary nulls (hard_wc under S2 and C0, EL-wide) with B = 2000 replications.
See src/el_wide_nulls.py. Requires RESULTS_ROOT/E30/E30_fit.json.

Outputs (RESULTS_ROOT/E38): E38_summary.json, E38_ckpt_{s2,c0}.json
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402
import el_wide_nulls  # noqa: E402

if __name__ == "__main__":
    out_dir = config.results_dir("E38")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(out_dir / "E38_run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    el_wide_nulls.main(mode="bigB", out_dir=out_dir, n_null_reps=2000, n_workers=config.n_workers(6))
