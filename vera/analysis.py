"""Claim table, scores, and evaluation.

Protocol (fixed before seeing test results):
  * Primary set: generated claims that assert an abnormal finding (not negated,
    not hedged), are localizable, and get a 0/1 label from the reference.
  * Risk = -alignment (low attention on the named region -> likely unsupported).
  * Every choice (layer, head subset, threshold) is made on the CALIBRATION
    split only; the TEST split is touched once for reporting.
  * CIs: 1000x bootstrap over images (claims from one image move together).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .anatomy import rect_masks, union
from .labels import label_claim, reference_index
from .text import extract_claims

FLAG_RATES = (0.1, 0.2, 0.3, 0.5)


# ---------------------------------------------------------------------------
# Building the claim table
# ---------------------------------------------------------------------------
def build_table(manifest, attn_dir, seg_dir=None, log=print):
    attn_dir = Path(attn_dir)
    seg_dir = Path(seg_dir) if seg_dir else None
    rows, Pl, Ph, Rm, VM, CF, Mr, Ms = [], [], [], [], [], [], [], []
    rect_cache = {}
    n_img = 0
    for it in manifest:
        mj, nz = attn_dir / f"{it['image_id']}.json", attn_dir / f"{it['image_id']}.npz"
        if not (mj.exists() and nz.exists()):
            continue
        meta, arr = json.loads(mj.read_text()), np.load(nz)
        G = meta["grid"]
        if G not in rect_cache:
            rect_cache[G] = rect_masks(G)
        rect = rect_cache[G]
        seg = None
        if seg_dir is not None and (seg_dir / f"{it['image_id']}.npz").exists():
            sz = np.load(seg_dir / f"{it['image_id']}.npz")
            if bool(sz["valid"]):
                seg = {k: sz[k] for k in sz.files if k != "valid"}
        ref = reference_index(it.get("findings", ""), it.get("impression", ""),
                              it.get("mesh", ""), it.get("problems", ""))
        span2s = {(a, b): i for i, (_, a, b) in enumerate(meta["sentences"])}
        n_img += 1
        lm = arr["layer_map"].astype(np.float32)
        hp = arr["head_pool"].astype(np.float32)
        rm = arr["report_map"].astype(np.float32)
        for c in extract_claims(meta["report"]):
            si = span2s.get((c.sent_start, c.sent_end))
            if si is None or not c.localizable:
                continue
            d = c.to_dict()
            d.update(label_claim(d, ref))
            d.update(image_id=it["image_id"], split=it["split"], ref_abnormal=it.get("ref_abnormal"),
                     seg_valid=seg is not None, sent_index=si)
            mr = union(rect, c.zones)
            ms = union(seg, c.zones) if seg is not None else None
            if mr is None:
                continue
            if ms is None or ms.sum() == 0:
                ms = mr
                d["seg_used"] = False
            else:
                d["seg_used"] = True
            L = lm.shape[1]
            p = lm[si].reshape(L, -1)
            Pl.append((p / np.clip(p.sum(1, keepdims=True), 1e-12, None)).astype(np.float16))
            h = hp[si].reshape(hp.shape[1], hp.shape[2], -1)
            Ph.append((h / np.clip(h.sum(-1, keepdims=True), 1e-12, None)).astype(np.float16))
            r = rm.reshape(L, -1)
            Rm.append((r / np.clip(r.sum(1, keepdims=True), 1e-12, None)).astype(np.float16))
            VM.append(arr["vis_mass"][si])
            CF.append(arr["conf"][si])
            Mr.append(mr.ravel())
            Ms.append(ms.ravel())
            rows.append(d)
    log(f"images: {n_img}, localizable claims: {len(rows)}")
    A = dict(Pl=np.stack(Pl).astype(np.float16), Ph=np.stack(Ph).astype(np.float16),
             Rm=np.stack(Rm).astype(np.float16), VM=np.stack(VM), CF=np.stack(CF),
             Mr=np.stack(Mr), Ms=np.stack(Ms)) if rows else {}
    return rows, A


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def boot_ci(y, s, groups, fn=auroc, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    ug = np.unique(groups)
    idx_by = {g: np.nonzero(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([idx_by[g] for g in pick])
        v = fn(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def at_flag_rate(y_te, r_te, r_cal, rate):
    thr = np.quantile(r_cal, 1 - rate)
    f = r_te >= thr
    tp = int((f & (y_te == 1)).sum())
    prec = tp / max(1, int(f.sum()))
    rec = tp / max(1, int(y_te.sum()))
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return dict(target_rate=rate, test_flag_rate=float(f.mean()), precision=prec, recall=rec, f1=f1)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def region_scores(P, M, prior=None):
    """P: [N, ..., K] normalised maps; M: [N, K] soft masks -> raw, lift, excess [N, ...]."""
    shp = (M.shape[0],) + (1,) * (P.ndim - 2) + (M.shape[1],)
    Mb = M.reshape(shp)
    raw = (P * Mb).sum(-1)
    area = M.mean(1).reshape((M.shape[0],) + (1,) * (P.ndim - 2))
    out = dict(raw=raw, lift=raw / np.clip(area, 1e-6, None))
    if prior is not None:
        out["excess"] = raw - (prior[None] * Mb).sum(-1)
    return out


def region_scores_chunked(P, M, prior=None, step=512):
    """region_scores in row chunks so float32 temporaries stay small."""
    outs = [region_scores(P[i:i + step].astype(np.float32), M[i:i + step], prior)
            for i in range(0, len(P), step)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def fit_probe(X, y, cal, te, groups, seed=0):
    """L2 logistic regression trained on calibration claims only; C picked by grouped CV
    on calibration. Returns risk for every row (test rows are never seen in fitting)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(X.reshape(len(X), -1).astype(np.float32))
    k = max(2, min(5, len(np.unique(groups[cal]))))
    cv = list(GroupKFold(k).split(X[cal], y[cal], groups[cal]))
    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                      {"logisticregression__C": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]},
                      cv=cv, scoring="roc_auc")
    gs.fit(X[cal], y[cal])
    return gs.predict_proba(X)[:, 1], float(gs.best_score_), gs.best_params_["logisticregression__C"]


def pool_mask(M, G, k=4):
    g = G // k
    return M.reshape(-1, g, k, g, k).mean((2, 4)).reshape(len(M), -1)


def entropy(P):
    P = np.clip(P, 1e-12, None)
    return -(P * np.log2(P)).sum(-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(rows, A, out_dir, n_boot=1000, top_k_heads=8, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(rows)
    lab = np.array([np.nan if r["label"] is None else r["label"] for r in rows], float)
    split = np.array([r["split"] for r in rows])
    imgs = np.array([r["image_id"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    neg = np.array([r["negated"] for r in rows])
    unc = np.array([r["uncertain"] for r in rows])
    tier = np.array([r["tier"] for r in rows])

    prim = (kind == "abnormal") & ~neg & ~unc & ~np.isnan(lab)
    cal, te = prim & (split == "calib"), prim & (split == "test")
    y = np.nan_to_num(lab).astype(int)
    Pl = A["Pl"]                              # [N, L, G*G] float16
    Ph = A["Ph"]                              # [N, L, H, g*g] float16
    G = int(round(np.sqrt(Pl.shape[-1])))
    L, H = Ph.shape[1], Ph.shape[2]

    # priors from calibration sentences only (any claim kind), per layer / head
    cal_any = split == "calib"
    prior_l = Pl[cal_any].astype(np.float32).mean(0)   # [L, K]
    prior_h = Ph[cal_any].astype(np.float32).mean(0)   # [L, H, k]

    feats = {}
    for mname, M in (("seg", A["Ms"]), ("rect", A["Mr"])):
        s = region_scores_chunked(Pl, M, prior_l)       # each [N, L]
        for met, v in s.items():
            feats[f"{mname}:{met}"] = v
        mp = pool_mask(M, G)
        sh = region_scores_chunked(Ph, mp, prior_h)     # [N, L, H]
        for met, v in sh.items():
            feats[f"{mname}:{met}:head"] = v

    results = dict(n_claims_total=N, n_primary_calib=int(cal.sum()), n_primary_test=int(te.sum()),
                   prevalence_calib=float(y[cal].mean()) if cal.any() else None,
                   prevalence_test=float(y[te].mean()) if te.any() else None,
                   n_images_test=int(len(np.unique(imgs[te]))), grid=G, n_layers=L, n_heads=H,
                   seg_used_fraction=float(np.mean([r.get("seg_used", False) for r, p in zip(rows, prim) if p])) if prim.any() else None)

    def report(name, risk, mask=te, cal_mask=cal):
        r_te, y_te = risk[mask], y[mask]
        rec = dict(name=name, auroc=auroc(y_te, r_te), auroc_ci=boot_ci(y_te, r_te, imgs[mask], n=n_boot),
                   auprc=auprc(y_te, r_te), n=int(mask.sum()))
        if cal_mask is not None and cal_mask.any():
            rec["calib_auroc"] = auroc(y[cal_mask], risk[cal_mask])
            rec["at_flag_rate"] = [at_flag_rate(y_te, r_te, risk[cal_mask], fr) for fr in FLAG_RATES]
        return rec

    # ---- layer selection on calibration (pre-specified direction: risk = -score)
    layer_curves, selected = {}, {}
    for key in ["seg:raw", "seg:lift", "seg:excess", "rect:raw", "rect:lift", "rect:excess"]:
        v = feats[key]
        cal_curve = [auroc(y[cal], -v[cal, l]) for l in range(L)]
        test_curve = [auroc(y[te], -v[te, l]) for l in range(L)]
        layer_curves[key] = dict(calib=cal_curve, test=test_curve)
        lbest = int(np.nanargmax(cal_curve))
        selected[key] = lbest
    results["layer_curves"] = layer_curves

    primary_key = "seg:lift"
    lsel = selected[primary_key]
    risk_primary = -feats[primary_key][:, lsel]
    results["primary"] = dict(config=f"{primary_key}, layer {lsel} (selected on calibration)",
                              **report("VERA (seg, lift, calib-selected layer)", risk_primary))

    # ---- head ensemble: top-k heads by calibration AUROC, averaged z-scores
    hk = feats[f"{primary_key}:head"]                                   # [N, L, H]
    flat = hk.reshape(N, -1)
    cal_h = np.array([auroc(y[cal], -flat[cal, j]) for j in range(flat.shape[1])])
    top = np.argsort(-np.nan_to_num(cal_h, nan=0))[:top_k_heads]
    mu, sd = flat[cal][:, top].mean(0), flat[cal][:, top].std(0) + 1e-9
    risk_heads = -((flat[:, top] - mu) / sd).mean(1)
    results["head_ensemble"] = dict(heads=[(int(j // H), int(j % H), float(cal_h[j])) for j in top],
                                    **report(f"VERA head ensemble (top {top_k_heads}, calib-selected)", risk_heads))

    # ---- calibration-trained probes (all fitting on calib claims; test touched once)
    # region: alignment of every head with the claim's own region.
    # shuffled: same features, but each claim gets another claim's region -> tests whether
    #           the NAMED region carries the signal (grounding) or just generic attention.
    # share: per-head attention share to the image, no region at all (non-spatial control).
    CF, VM = A["CF"], A["VM"]
    Ms_pool = pool_mask(A["Ms"], G)
    perm = np.random.default_rng(1).permutation(N)
    sh_shuf = region_scores_chunked(Ph, Ms_pool[perm], prior_h)
    X_region = np.concatenate([feats["seg:raw:head"].reshape(N, -1), feats["seg:lift:head"].reshape(N, -1),
                               feats["seg:excess:head"].reshape(N, -1)], 1)
    X_shuf = np.concatenate([sh_shuf["raw"].reshape(N, -1), sh_shuf["lift"].reshape(N, -1),
                             sh_shuf["excess"].reshape(N, -1)], 1)
    X_share = np.concatenate([VM.reshape(N, -1), CF], 1)
    probes = []
    for nm, X in (("Probe: region alignment, all heads (VERA-probe)", X_region),
                  ("Probe: region alignment with SHUFFLED regions (control)", X_shuf),
                  ("Probe: image-attention share + confidence, no region (control)", X_share),
                  ("Probe: region alignment + share + confidence", np.concatenate([X_region, X_share], 1))):
        if cal.sum() < 20:
            break
        risk_p, cv_auc, C = fit_probe(X, y, cal, te, imgs)
        rec = report(nm, risk_p)
        rec.update(calib_cv_auroc=cv_auc, C=C)
        probes.append(rec)
        if nm.startswith("Probe: region alignment, all heads"):
            risk_probe = risk_p
    results["probes"] = probes

    # ---- baselines on the same claims
    lay = lsel
    base = [
        report("Confidence: mean token log-prob", -CF[:, 0]),
        report("Confidence: min token log-prob", -CF[:, 1]),
        report("Visual reliance (attention share to image, all layers/heads)", -VM.mean((1, 2))),
        report(f"Attention entropy (layer {lay})", entropy(Pl[:, lay].astype(np.float32))),
    ]
    # v1 recipe at the corrected grid: rect masks, raw score, last 4 layers, report-level map
    Rm = A["Rm"].astype(np.float32)
    rep_raw = region_scores(Rm[:, -4:].mean(1, keepdims=True), A["Mr"])["raw"][:, 0]
    base.append(report("v1 recipe (rect, raw, last-4 layers, whole-report map)", -rep_raw))
    rng = np.random.default_rng(0)
    base.append(report("Random", rng.random(N)))
    results["baselines"] = base
    results["flag_all_f1_test"] = float(2 * y[te].mean() / (1 + y[te].mean())) if te.any() else None

    # ---- ablations (test AUROC; layer chosen on calibration for each variant)
    abl = []
    for key in ["seg:raw", "seg:lift", "seg:excess", "rect:raw", "rect:lift", "rect:excess"]:
        abl.append(report(f"{key} @ calib layer {selected[key]}", -feats[key][:, selected[key]]))
    last4 = region_scores(Pl[:, -4:].mean(1, keepdims=True), A["Ms"])["lift"][:, 0]
    abl.append(report("seg:lift, last-4-layer average (v1 layer choice)", -last4))
    rep_lift = region_scores(Rm[:, lsel:lsel + 1], A["Ms"])["lift"][:, 0]
    abl.append(report(f"seg:lift, whole-report map instead of sentence (layer {lsel})", -rep_lift))
    results["ablations"] = abl

    # ---- per tier, and secondary claim types
    results["per_tier"] = {t: report(f"primary, tier={t}", risk_primary, mask=te & (tier == t), cal_mask=None)
                           for t in ("critical", "moderate", "mild")}
    sec = {}
    lv = ~np.isnan(lab)
    for nm, m in (("negated_claims", (kind == "abnormal") & neg & lv),
                  ("normal_claims", (kind == "normal") & lv),
                  ("hedged_claims", (kind == "abnormal") & ~neg & unc & lv)):
        sec[nm] = report(nm, risk_primary, mask=m & (split == "test"), cal_mask=None)
    results["secondary"] = sec
    results["label_reasons_primary"] = {k: int(v) for k, v in zip(*np.unique(
        [r["label_reason"] for r, p in zip(rows, prim) if p], return_counts=True))} if prim.any() else {}

    _figures(results, feats, y, te, risk_primary, base, out_dir,
             risk_probe=risk_probe if probes else None)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    _write_claims_csv(rows, risk_primary, out_dir / "claims.csv")
    (out_dir / "summary.md").write_text(summary_md(results))
    log(summary_md(results))
    return results, risk_primary


def _write_claims_csv(rows, risk, path):
    import csv
    keys = ["image_id", "split", "sentence", "kind", "finding", "tier", "negated", "uncertain",
            "relational", "location_text", "zones", "label", "label_reason", "seg_used"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["vera_risk"])
        for r, s in zip(rows, risk):
            w.writerow([("|".join(r[k]) if k == "zones" else r.get(k)) for k in keys] + [float(s)])


def _fmt(r):
    ci = r.get("auroc_ci", [np.nan, np.nan])
    return f"{r['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def summary_md(R):
    lines = ["# VERA v3 results", "",
             f"- Primary claims: calib {R['n_primary_calib']}, test {R['n_primary_test']} "
             f"({R['n_images_test']} test images); hallucination prevalence test = {R['prevalence_test']}",
             f"- Visual grid {R['grid']}x{R['grid']}, {R['n_layers']} layers x {R['n_heads']} heads; "
             f"segmentation masks used for {R['seg_used_fraction']} of primary claims",
             f"- Flag-all F1 on test: {R['flag_all_f1_test']}", "",
             "## Primary (pre-specified)", f"- {R['primary']['config']}: AUROC {_fmt(R['primary'])}, "
             f"calib AUROC {R['primary'].get('calib_auroc', float('nan')):.3f}",
             f"- Head ensemble: AUROC {_fmt(R['head_ensemble'])}", "",
             "## Calibration-trained probes (test AUROC, 95% CI)"]
    lines += [f"- {p['name']}: {_fmt(p)} (calib CV {p['calib_cv_auroc']:.3f}, C={p['C']})" for p in R.get("probes", [])]
    lines += ["", "## Baselines (test AUROC, 95% CI)"]
    lines += [f"- {b['name']}: {_fmt(b)}" for b in R["baselines"]]
    lines += ["", "## Ablations (test AUROC, 95% CI)"] + [f"- {a['name']}: {_fmt(a)}" for a in R["ablations"]]
    lines += ["", "## Per tier"] + [f"- {k}: {_fmt(v)} (n={v['n']})" for k, v in R["per_tier"].items()]
    lines += ["", "## Operating points (primary, thresholds from calibration)"]
    for op in R["primary"].get("at_flag_rate", []):
        lines.append(f"- target flag rate {op['target_rate']:.0%}: test flag rate {op['test_flag_rate']:.1%}, "
                     f"recall {op['recall']:.3f}, precision {op['precision']:.3f}, F1 {op['f1']:.3f}")
    lines += ["", "## Secondary claim types (same score, test)"] + [f"- {k}: {_fmt(v)} (n={v['n']})" for k, v in R["secondary"].items()]
    lines += ["", f"Primary label reasons: {R['label_reasons_primary']}"]
    return "\n".join(lines)


def _figures(R, feats, y, te, risk_primary, base, out_dir, risk_probe=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, c in [("seg:lift", "C0"), ("seg:excess", "C1"), ("rect:lift", "C2")]:
        cur = R["layer_curves"][key]
        ax[0].plot(cur["calib"], color=c, label=f"{key} calib")
        ax[0].plot(cur["test"], color=c, ls="--", alpha=.6, label=f"{key} test")
    ax[0].axhline(.5, color="k", lw=.8)
    ax[0].set_xlabel("decoder layer")
    ax[0].set_ylabel("AUROC")
    ax[0].legend(fontsize=7)
    ax[0].set_title("AUROC by layer")

    def roc(yv, s):
        o = np.argsort(-s)
        yv = yv[o]
        tpr = np.cumsum(yv) / max(1, yv.sum())
        fpr = np.cumsum(1 - yv) / max(1, (1 - yv).sum())
        return np.r_[0, fpr], np.r_[0, tpr]

    fx, fy = roc(y[te], risk_primary[te])
    ax[1].plot(fx, fy, lw=2, label=f"VERA training-free ({R['primary']['auroc']:.3f})")
    if risk_probe is not None:
        px, py = roc(y[te], risk_probe[te])
        ax[1].plot(px, py, lw=2, label=f"VERA-probe ({R['probes'][0]['auroc']:.3f})")
    ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
    ax[1].set_xlabel("FPR")
    ax[1].set_ylabel("TPR")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Test ROC (primary)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_layers_roc.png", dpi=160)
    plt.close(fig)
