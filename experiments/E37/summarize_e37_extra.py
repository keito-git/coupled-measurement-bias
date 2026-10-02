"""
Additional E37 summaries: mean extrapolation path over lambda = 0, 1, 2, 3, number of replications without
a converged estimate at every lambda, and the share of events whose fractional matrix powers
(lambda = 0.5, 1.5) are not stochastic (one simulated data set at gamma = 0, rep 0).
Inputs: E37_results.json, pi_table.npy. Output: E37_extra_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import run_E37 as m  # noqa: E402
import latent_mark as d  # noqa: E402
from latent_mark import simulate_matched as simulate  # noqa: E402

HERE = config.results_dir("E37")
res = json.loads((HERE / "E37_results.json").read_text())
rows = res["rows"]
out = {"path_mean": {}, "n_excluded": {}}
for g in (0.0, -0.4):
    P = np.array([[r["path"][k] for k in ("0.0", "1.0", "2.0", "3.0")] for r in rows if r["gamma"] == g], dtype=float)
    out["path_mean"][str(g)] = [float(x) for x in np.nanmean(P, 0)]
    out["n_excluded"][str(g)] = int(np.isnan(P[:, 3]).sum())
lens, mu, alpha, beta = d.fit_el_wide_generator(n_restarts=1)
ann = d.load_annotator_params()
T = np.load(HERE / "pi_table.npy")
a, b = ann["a"], ann["b"]
conf = np.asarray(ann["confusion"], float); prior = np.asarray(ann["prior"], float)
dlgs = simulate(0, 0.0, lens, mu, alpha, float(beta), ann)
n_ev = n_bad = 0
for dlg in dlgs:
    for x in dlg:
        q = 1 / (1 + np.exp(-(a - b * m.post_mean_u(x["votes"], a, b, conf, prior, d.N_ANN_EL))))
        P = m.pi_for_q(T, q)
        bad = any(m.fractional_matrix_power(P, lam)[1] > 0 for lam in (0.5, 1.5))
        n_ev += 1; n_bad += int(bad)
out["frac_events_nonstochastic_fractional"] = n_bad / n_ev
out["n_events_checked"] = n_ev
(HERE / "E37_extra_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
