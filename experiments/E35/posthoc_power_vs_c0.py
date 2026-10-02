"""
Size-matched power of the hard_wc diagnostic (not pre-registered). The power replications of E35 (Scheme C'
generator at true gamma != 0) are tested against the C0 band, i.e. the null band of the same generator at
gamma = 0, so that the rejection rate does not mix power with simulator mismatch.
Inputs: E35_ckpt_power.json, E35_ckpt_c0_null.json. Output: posthoc_power_vs_c0.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

RUN = config.results_dir("E35")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


c0 = json.loads((RUN / "E35_ckpt_c0_null.json").read_text())["results"]
g0 = np.array([x["fits"]["hard_wc"]["gamma_hat"] for x in c0])
lo, hi = np.quantile(g0, [0.025, 0.975])
power = json.loads((RUN / "E35_ckpt_power.json").read_text())["results"]
by = defaultdict(list)
for x in power:
    by[x["gamma_inj"]].append(x["hard_wc"]["gamma_hat"])
rows = []
for g in sorted(by):
    a = np.array(by[g])
    k = int(((a < lo) | (a > hi)).sum())
    rows.append({"gamma_true": g, "n": len(a), "n_reject": k, "rate": k / len(a), "wilson95": wilson(k, len(a)),
                 "gamma_hat_mean": float(a.mean()), "gamma_hat_sd": float(a.std(ddof=1))})
out = {"note": __doc__.strip(), "c0_band": [float(lo), float(hi)], "c0_n": int(len(g0)),
       "c0_mean": float(g0.mean()), "c0_sd": float(g0.std(ddof=1)), "rows": rows}
(RUN / "posthoc_power_vs_c0.json").write_text(json.dumps(out, indent=1))
for r in rows:
    print(r["gamma_true"], r["n_reject"], r["n"], round(r["rate"], 2), [round(v, 3) for v in r["wilson95"]],
          round(r["gamma_hat_mean"], 3))
