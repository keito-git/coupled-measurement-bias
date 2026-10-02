"""
Two-sided empirical p-values for the null-band comparisons of E27e (Scheme A') and E33c.
Definition: p = (1 + #{|g_b - med| >= |g_real - med|}) / (B + 1), med = null median.
The decision rule (real estimate outside the 2.5-97.5% band of the null) is unchanged. Per-replication null
values are read from the experiment checkpoints; nothing is re-simulated.
Output: analysis/two_sided_pvalues.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import config  # noqa: E402

X = config.RESULTS_ROOT


def p2(null, real):
    null = np.asarray([v for v in null if v is not None and np.isfinite(v)])
    med = np.median(null)
    k = int((np.abs(null - med) >= abs(real - med)).sum())
    p = (1 + k) / (len(null) + 1)
    return {"p": p, "mc_se": float(np.sqrt(p * (1 - p) / len(null))), "B": int(len(null)),
            "q2_5": float(np.quantile(null, 0.025)), "q97_5": float(np.quantile(null, 0.975)),
            "reject_band": bool(not (np.quantile(null, 0.025) <= real <= np.quantile(null, 0.975)))}


out = {"definition": __doc__.strip()}

# E27e Scheme A' (hard marks; indep/linked; pl1/3/4; within-cell and raw H)
e = json.loads((X / "E27e/E27e_results.json").read_text())
reps = json.loads((X / "E27e/E27e_partial_schemeA.json").read_text())["scheme_a_prime"]
real_wc = {1: e["real_gamma"]["full"], 3: e["real_gamma"]["p3_5"], 4: e["real_gamma"]["p4_5"]}
e35 = json.loads((X / "E35/E35_results.json").read_text())
real_rh_pl1 = e35["real_gammas"]["hard_rh"]
e27 = {}
for model in ("indep", "linked"):
    for pl in (1, 3, 4):
        vals = [r["results"][model][str(pl)]["wc"]["gamma_hat"] for r in reps]
        e27[f"{model}_pl{pl}_wc"] = p2(vals, real_wc[pl])
    vals = [r["results"][model]["1"]["rh"]["gamma_hat"] for r in reps]
    e27[f"{model}_pl1_rh"] = p2(vals, real_rh_pl1)
out["E27e_schemeA"] = e27
out["E27e_n_reps"] = len(reps)

# E33c per-corpus diagnostic: S1 from the E33c checkpoints, S2 = rep ids 0..199 of the E40 checkpoints
e33 = json.loads((X / "E33c/E33c_results.json").read_text())
n_s1 = e33["config"]["n_reps"]
diag = {}
for c in ("friends", "emotionpush", "m3ed"):
    real = e33["real_gammas"][c]["gamma_hat"]
    s1 = json.loads((X / f"E33c/E33c_ckpt_{c}_S1.json").read_text())["results"]
    s2 = [r for r in json.loads((X / f"E40/E40_ckpt_{c}_S2.json").read_text())["results"] if r["rep"] < n_s1]
    for sim, rr in (("S1", s1), ("S2", s2)):
        vals = [r["gamma_hat"] for r in rr if r.get("converged", True)]
        d = p2(vals, real)
        # consistency with the stored decision (same band rule)
        assert d["reject_band"] == e33["decisions"][f"{c}_{sim}"]["reject_gamma0"], (c, sim)
        diag[f"{c}_{sim}"] = d
out["E33c"] = diag

(HERE / "two_sided_pvalues.json").write_text(json.dumps(out, indent=1))
for grp in ("E27e_schemeA", "E33c"):
    for k, v in out[grp].items():
        print(grp, k, f"p={v['p']:.3f}", "reject" if v["reject_band"] else "")
