"""Offline end-to-end test of claims -> labels -> masks -> scores -> evaluation.
Uses the 50 real CheXagent reports + their IU references, replicated, with fake
attention. With a planted signal the pipeline must find it; without, ~0.5."""
import json, glob, sys, tempfile, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vera.text import split_sentences, extract_claims
from vera.anatomy import rect_masks, union
from vera.analysis import build_table, evaluate

SRC = Path("/home/claude/proj/VERA_GENAI4HEALTH/NEW_OUTPUT")
ref = {}
for sp in ["val", "test"]:
    for e in json.load(open(SRC / f"processed/{sp}.json")):
        ref[e["image_id"]] = e["report"]
pairs = [(json.load(open(f))["generated_report"], ref[json.load(open(f))["image_id"]])
         for f in sorted(glob.glob(str(SRC / "attention_maps/chexagent/*.dcm_meta.json")))]

def make(work, plant, n=400, G=32, L=8, H=4, seed=0):
    rng = np.random.default_rng(seed); random.seed(seed)
    from vera.labels import reference_index, label_claim
    rect = rect_masks(G); items = []
    for k in range(n):
        gen, rtxt = pairs[k % len(pairs)]
        iid = f"img{k}"
        sents = split_sentences(gen)
        S = len(sents)
        lm = rng.gamma(0.5, size=(S, L, G, G)).astype(np.float32)
        idx = reference_index(rtxt, "")
        for c in extract_claims(gen):
            if plant and c.localizable and c.kind == "abnormal" and not c.negated:
                lab = label_claim(c.to_dict(), idx)["label"]
                si = [i for i, s in enumerate(sents) if s.start == c.sent_start][0]
                if lab == 0:  # supported claims look at their region in layer 5
                    lm[si, 5] += 8 * union(rect, c.zones) * lm[si, 5].mean()
        hp = lm.reshape(S, L, 1, 8, 4, 8, 4).sum((4, 6)).repeat(H, 2)
        np.savez_compressed(work / "attention" / f"{iid}.npz", layer_map=lm.astype(np.float16),
                            head_pool=hp.astype(np.float16), vis_mass=rng.random((S, L, H)).astype(np.float32),
                            conf=rng.normal(size=(S, 2)).astype(np.float32),
                            report_map=lm.mean(0).astype(np.float16), token_logprob=np.zeros(5, np.float32))
        meta = dict(image_path=str(work / "img.png"), report=gen, grid=G, n_layers=L, n_heads=H, n_tokens=5,
                    sentences=[[s.text, s.start, s.end] for s in sents], sentence_token_spans=[[0, 1]] * S)
        (work / "attention" / f"{iid}.json").write_text(json.dumps(meta))
        items.append(dict(image_id=iid, findings=rtxt, impression="", split="calib" if k < n * .3 else "test"))
    return items

for plant in (True, False):
    with tempfile.TemporaryDirectory() as d:
        w = Path(d); (w / "attention").mkdir()
        from PIL import Image; Image.fromarray((np.random.rand(600, 500) * 255).astype("uint8")).save(w / "img.png")
        items = make(w, plant)
        rows, A = build_table(items, w / "attention", None, log=lambda *a: None)
        R, risk = evaluate(rows, A, w / "res", n_boot=200, log=lambda *a: None)
        p = R["primary"]
        print(f"plant={plant}: test claims {R['n_primary_test']}, AUROC {p['auroc']:.3f} CI {np.round(p['auroc_ci'],3)}, "
              f"selected: {p['config']}, rect:lift layer-5 test AUROC {R['layer_curves']['rect:lift']['test'][5]:.3f}")
        assert (p["auroc"] > 0.8) if plant else (0.35 < p["auroc"] < 0.65)
        pr = {q["name"].split(" (")[0].split(",")[0]: q["auroc"] for q in R["probes"]}
        print("   probes:", {k: round(v, 3) for k, v in pr.items()})
        reg, shuf = R["probes"][0]["auroc"], R["probes"][1]["auroc"]
        if plant:
            assert reg > 0.8 and reg - shuf > 0.15, "probe must find the signal, shuffled regions must lose it"
        else:
            assert 0.3 < reg < 0.7
        assert (w / "res" / "fig_layers_roc.png").exists() and (w / "res" / "claims.csv").exists()
        from vera.review import make_review_pack
        make_review_pack(rows, A, risk, w / "res" / "review", w / "attention", n_each=5)
        assert len(list((w / "res" / "review" / "overlays").glob("*.png"))) == 10
print("synthetic test passed")
