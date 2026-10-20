"""Mock CheXagent: checks image-span detection (32x32), token->sentence alignment,
aggregation shapes and that spatial layout survives (a planted hot patch must land
at the right (row, col) on the 32x32 grid)."""
import sys, types
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vera.extract import extract_one

TEXT = "[Breathing: Lungs] **There is a calcified granuloma in the right lower lobe.** [Cardiac: Heart] The heart size is normal."
WORDS = TEXT.split(" ")
PIECES = [w + (" " if i < len(WORDS) - 1 else "") for i, w in enumerate(WORDS)]   # one token per word
START, END, PAD = 1000, 1001, 1002
NV, PRE, POST = 1024, 5, 7

class Tok:
    img_start_id, img_end_id = START, END
    def from_list_format(self, x): return "q"
    def apply_chat_template(self, conv, **k): return "prompt"
    def __call__(self, text, return_tensors):
        ids = list(range(PRE)) + [START] + [PAD] * NV + [END] + list(range(POST))
        o = types.SimpleNamespace(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids)))
        o.to = lambda dev: o
        return o
    def decode(self, ids, skip_special_tokens=True):
        return "".join(PIECES[i - 2000] for i in ids)

class Model(torch.nn.Module):
    def __init__(s): super().__init__(); s.w = torch.nn.Parameter(torch.zeros(1))
    def generate(s, input_ids, attention_mask, max_new_tokens, **k):
        S0, T, L, H = input_ids.shape[1], len(PIECES), 3, 2
        gen = torch.arange(2000, 2000 + T)[None]
        atts, scores = [], []
        for t in range(T):
            q = S0 if t == 0 else 1
            layers = []
            for l in range(L):
                a = torch.rand(1, H, q, S0 + t) * 0.01
                # hot patch at grid (row 20, col 5) for every token of the first sentence
                if t < 12: a[0, :, -1, PRE + 1 + 20 * 32 + 5] = 5.0
                a = a / a.sum(-1, keepdim=True)
                layers.append(a)
            atts.append(tuple(layers))
            scores.append(torch.randn(1, 3000))
        return types.SimpleNamespace(sequences=torch.cat([input_ids, gen], 1), attentions=tuple(atts), scores=tuple(scores))

arrays, meta = extract_one(Model(), Tok(), "img.png", 256)
print("grid", meta["grid"], "sentences", [s[0] for s in meta["sentences"]], "token spans", meta["sentence_token_spans"])
assert meta["grid"] == 32 and len(meta["sentences"]) == 2
lm = arrays["layer_map"].astype(np.float32)
r, c = np.unravel_index(lm[0, 0].argmax(), (32, 32))
print("hot patch recovered at", (r, c), "| layer_map", lm.shape, "| head_pool", arrays["head_pool"].shape, "| vis_mass", arrays["vis_mass"].shape)
assert (r, c) == (20, 5)
print("vis_mass sent0", arrays["vis_mass"][0].round(3).tolist()); assert (arrays["vis_mass"] > 0.9).all() and (arrays["vis_mass"] <= 1.0 + 1e-3).all()
print("mock extraction test passed")
