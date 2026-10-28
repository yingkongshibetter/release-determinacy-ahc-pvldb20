"""Summary of the batch audits of scaling/batch_audit.py: release build time and size, request volumes,
batch times, and the check of every audit. Writes outputs/scaling/batch_summary.txt.

usage: python scaling/batch_summary.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

S = str(C.OUT) + "/"
DS = [d for d in C.DATASETS if (C.OUT / "batch" / f"{d}_meta.json").exists()]
out = []
tot = fc = fcT = same = 0
rows = []
for ds in DS:
    au = pd.read_csv(S + f'batch/{ds}_audits.csv')
    pr = pd.read_csv(S + f'batch/{ds}_pairs.csv')
    meta = json.load(open(S + f'batch/{ds}_meta.json', encoding='utf8'))
    bt = pd.read_csv(S + f'batch/{ds}_batch.csv').median(numeric_only=True)
    tot += len(au)
    fc += int(((au.cert_P == 1) & (au.pass_exact == 0)).sum())
    fcT += int((((au.cert_T8 == 1) | (au.cert_T32 == 1)) & (au.pass_exact == 0)).sum())
    same += int((au.verdict_a == au.verdict_b).sum())
    b = pr[pr.is_base == 1]
    ab = au[au.is_base == 1]
    f1 = ab[abs(ab.factor - 1.0) < 1e-12]
    rows.append(dict(ds=ds, audits_all=len(au), pairs_all=len(pr), base_pairs=len(b),
                     scan_ms_med_base=1e3 * b.t_scan_P_s.median(), scan_ms_min_base=1e3 * b.t_scan_P_s.min(), scan_ms_max_base=1e3 * b.t_scan_P_s.max(),
                     exact_ms_med_base=1e3 * b.t_exact_s.median(), exact_ms_min_base=1e3 * b.t_exact_s.min(), exact_ms_max_base=1e3 * b.t_exact_s.max(),
                     scan_ms_med_all=1e3 * pr.t_scan_P_s.median(), exact_ms_med_all=1e3 * pr.t_exact_s.median(),
                     P_build_s=meta['t_build_P_s'], T8_build_s=meta['t_build_P_s'] + meta['t_build_T8_from_P_s'],
                     T32_build_s=meta['t_build_P_s'] + meta['t_build_T32_from_P_s'],
                     P_MB=meta['bytes_release_P'] / 1e6, T8deg_MB=meta['bytes_release_T8_deg'] / 1e6, T32deg_MB=meta['bytes_release_T32_deg'] / 1e6,
                     a_s=bt.a_total_s, a_fused_s=bt.af_total_s, b_s=bt.b_total_s, b_shared_s=bt.bs_total_s,
                     a_plus_build_s=bt.a_total_s + meta['t_build_P_s'],
                     base_open_pairs_any_factor=int(b.open_P.sum()),
                     base_requests_at_B=int((f1.cert_P == 0).sum()), base_cert_at_B=int(f1.cert_P.sum()),
                     agg_bytes_per_pair=meta['bytes_aggregates_per_pair'], n=meta['n'], n_sizes=meta['n_sizes']))
R = pd.DataFrame(rows)
# bytes moved by requests at the shared budget (factor 1.0) and over the whole base workload (any factor)
for i, ds in enumerate(DS):
    au = pd.read_csv(S + f'batch/{ds}_audits.csv')
    pr = pd.read_csv(S + f'batch/{ds}_pairs.csv').set_index('pair_idx')
    ab = au[au.is_base == 1]
    for tag, sel in (('B', ab[abs(ab.factor - 1.0) < 1e-12]), ('any', ab)):
        op = sorted(sel[sel.cert_P == 0].pair_idx.unique())
        sub = pr.loc[op] if op else pr.iloc[:0]
        cols = sub.attribute.nunique() + sub.part_id.nunique()
        R.loc[i, f'req_pairs_{tag}'] = len(op)
        R.loc[i, f'req_cols_MB_{tag}'] = cols * R.loc[i, 'n'] * 4 / 1e6
        R.loc[i, f'req_agg_KB_{tag}'] = len(op) * R.loc[i, 'agg_bytes_per_pair'] / 1024
out.append(R.T.to_string())
out.append(f"producer release build for methods besides AHRC and AHCKA (P): at most {R.P_build_s.max():.2f} s and "
           f"{R.P_MB.max():.0f} MB")
out.append(f"requests per dataset over the base workload: at most {R.req_cols_MB_any.max():.2f} MB of attribute and "
           f"partition columns sent, at most {R.req_agg_KB_any.max():.1f} KB of per-size counts and sums returned")
out.append(f"certify-then-request with the release build: at most {R.a_plus_build_s.max():.2f} s per dataset; "
           f"exact evaluation of every audit: at most {R.b_s.max():.2f} s per dataset")
out.append(f'ALL audits: {tot}; false certificates U_P: {fc}; false certificates T8/T32: {fcT}; verdict (a)==(b): {same} of {tot}')
txt = '\n'.join(out)
R.to_csv(S + 'batch_summary.csv', index=False)
open(S + 'batch_summary.txt', 'w', encoding='utf8').write(txt + '\n')
print(txt)
