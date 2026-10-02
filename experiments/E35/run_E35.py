"""
E35: full decision table (4 estimators x S1/S2/C0 nulls, B = 200), power of the hard_wc diagnostic
under the Scheme C' generator, cross-simulator type-I error, and the S1' entropy-matched sensitivity check.
See src/el_wide_nulls.py. Requires RESULTS_ROOT/E30/E30_fit.json.

Outputs (RESULTS_ROOT/E35): E35_results.json, E35_ckpt_{s1,s2,c0,s1prime}_null.json, E35_ckpt_power.json
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402
import el_wide_nulls  # noqa: E402

if __name__ == "__main__":
    out_dir = config.results_dir("E35")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(out_dir / "E35_run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    el_wide_nulls.main(mode="table", out_dir=out_dir, n_null_reps=200, n_workers=config.n_workers(6))
