"""Summary of scaling/scale_up.py: one row per configuration and the log-log slopes of time against memberships.

usage: python scaling/scale_up_summary.py      -> outputs/scaling/scale_up/scale_up_table.csv, scale_up_slopes.csv
The paper reads the slope of t_exact_s over the imdb_cast_full copies up to x8 (no spill) and over all copies.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import np, pd  # noqa: E402

R = C.OUT / "scale_up"
# ---------------- S2
ph = {}
for p in ("load", "exact", "release", "fullP", "scan"):
    f = R / f"{p}.csv"
    if f.exists():
        ph[p] = pd.read_csv(f).drop_duplicates("config", keep="last").set_index("config")
rows = []
for cfg in ph["load"].index:
    L = ph["load"].loc[cfg]
    r = {"config": cfg, "source": L.source, "replicas": int(L.replicas), "memberships": int(L.memberships),
         "n": int(L.n), "m": int(L.m), "sum_s2": int(L.sum_s2), "db_bytes": int(L.db_bytes),
         "raw_incidence_bytes": int(L.raw_incidence_bytes), "t_load_s": L.t_s}
    if "exact" in ph and cfg in ph["exact"].index:
        E = ph["exact"].loc[cfg]
        r.update(t_exact_s=E.t_s_run0, t_exact_run1_s=E.t_s_run1, t_label_s=E.t_label_s_run0,
                 peak_rss_exact=E.peak_rss_run0, peak_temp_exact=E.peak_temp_run0,
                 hgp_equals_reference=int(E.HGP_equals_reference))
    if "release" in ph and cfg in ph["release"].index:
        X = ph["release"].loc[cfg]
        r.update(t_release_T8_s=X.t_s, release_chunks=int(X.chunks), peak_rss_release=X.peak_rss,
                 peak_temp_release=X.peak_temp, T8_rows=int(X.T_rows), bytes_release_T8_deg=int(X.bytes_release_T_deg))
    if "fullP" in ph and cfg in ph["fullP"].index:
        F = ph["fullP"].loc[cfg]
        r.update(t_release_P_s=F.t_s, peak_rss_P=F.peak_rss, peak_temp_P=F.peak_temp, P_rows_upper=int(F.P_rows_upper),
                 bytes_P=int(F.bytes_P), t_scan_P_s=F.t_scan_s)
    if "scan" in ph and cfg in ph["scan"].index:
        S = ph["scan"].loc[cfg]
        r.update(t_closure_s=S.t_closure_s, t_scan_T8_s=S.t_scan_s_run0, t_scan_T8_run1_s=S.t_scan_s_run1,
                 t_scan_T8_direct_s=S.t_scan_direct_s, peak_rss_scan=S.peak_rss, peak_temp_scan=S.peak_temp,
                 bound_T8=S.bound_T, U_P_ref=S.get("U_P_ref"), HGP_ref=S.get("HGP_ref"),
                 bound_ge_UP=S.get("bound_ge_UP"))
    rows.append(r)
s2 = pd.DataFrame(rows).sort_values("memberships")
s2.to_csv(R / "scale_up_table.csv", index=False)


def slope(x, y):
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log10(x[m]), np.log10(y[m])
    b, a = np.polyfit(lx, ly, 1)
    r2 = 1 - ((ly - (a + b * lx)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return b, r2


rep = s2[s2.source == "imdb_cast_full"]
fits = []
for col in ("t_exact_s", "t_release_T8_s", "t_closure_s", "t_scan_T8_s", "t_release_P_s", "t_scan_P_s"):
    if col in s2:
        for name, df in (("all_configs", s2), ("imdb_full_replicas", rep), ("imdb_full_replicas_le_x8", rep[rep.replicas <= 8])):
            b, r2 = slope(df.memberships.to_numpy(float), df[col].to_numpy(float))
            fits.append({"metric": col, "points": name, "n_points": int(df[col].notna().sum()), "slope": b, "r2": r2})
pd.DataFrame(fits).to_csv(R / "scale_up_slopes.csv", index=False)
print(s2.to_string())
print(pd.DataFrame(fits).to_string())

