# VERA v6

- 981 images; 1355 greedy primary claims; 5984 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 1360, 'A_s': 1360, 'T': 73, 'C': 12, 'S': 5, 'Q': 3, 'V': 21, 'I': 272, 'Aoff': 1360, 'Aoff_shuf': 1360}

## strict labels (n=1355, prevalence 0.210)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.682 [0.644, 0.719] | 0.682 [0.629, 0.730] | 0.675 [0.624, 0.722] |
| Confidence at the finding word | baseline | 0.614 [0.578, 0.647] | 0.601 [0.552, 0.651] | 0.600 [0.554, 0.646] |
| Self-consistency (3 sampled reports) | baseline | 0.669 [0.628, 0.706] | 0.640 [0.586, 0.691] | 0.647 [0.596, 0.697] |
| Self-verification P(yes), no training | baseline | 0.557 [0.517, 0.597] | 0.543 [0.487, 0.597] | 0.543 [0.487, 0.597] |
| External classifier (not trained on IU) | baseline | 0.669 [0.632, 0.704] | 0.675 [0.625, 0.723] | 0.648 [0.601, 0.696] |
| Image-level attention (control) | control | 0.687 [0.651, 0.725] | 0.675 [0.624, 0.721] | 0.669 [0.615, 0.720] |
| VERA-probe, offline region features | control-pair | 0.773 [0.736, 0.805] | 0.758 [0.714, 0.801] | 0.744 [0.698, 0.788] |
| VERA-probe, SHUFFLED regions (control) | control | 0.745 [0.707, 0.779] | 0.733 [0.686, 0.779] | 0.724 [0.673, 0.773] |
| VERA-probe (finding-word attention) | method | 0.774 [0.738, 0.807] | 0.759 [0.715, 0.801] | 0.745 [0.699, 0.789] |
| VERA-probe + sampled-report training | method | 0.809 [0.780, 0.839] | 0.798 [0.756, 0.836] | 0.762 [0.717, 0.805] |
| **VERA (full, model-internal)** | headline | 0.814 [0.783, 0.842] | 0.798 [0.757, 0.835] | 0.768 [0.726, 0.809] |
| VERA (full) + external classifier | method | 0.816 [0.785, 0.845] | 0.808 [0.766, 0.845] | 0.787 [0.741, 0.828] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.092 [+0.060, +0.122], p=0.001 | +0.070 [+0.036, +0.103], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.160 [+0.121, +0.199], p=0.001 | +0.145 [+0.098, +0.190], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.027 [+0.006, +0.048], p=0.005 | +0.021 [-0.014, +0.055], p=0.116 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.087 [+0.049, +0.124], p=0.001 | +0.076 [+0.018, +0.136], p=0.006 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.035 [+0.019, +0.053], p=0.001 | +0.017 [+0.000, +0.033], p=0.023 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.004 [-0.011, +0.017], p=0.273 | +0.007 [-0.014, +0.026], p=0.281 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.145 [+0.109, +0.183], p=0.001 | +0.121 [+0.078, +0.161], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.257 [+0.208, +0.300], p=0.001 | +0.226 [+0.152, +0.296], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.131 [+0.096, +0.164], p=0.001 | +0.093 [+0.060, +0.126], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | +0.002 [-0.013, +0.017], p=0.394 | +0.019 [-0.002, +0.042], p=0.035 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.145 [+0.103, +0.186], p=0.001 | +0.120 [+0.065, +0.176], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.610, recall 0.291; flag 20%: precision 0.542, recall 0.516; flag 30%: precision 0.462, recall 0.660; flag 50%: precision 0.363, recall 0.863

Headline per tier (5-fold CV): critical 0.658 [0.492, 0.809] (n=94); moderate 0.834 [0.805, 0.862] (n=1197); mild 0.546 [0.379, 0.724] (n=64)

## broad labels (n=1355, prevalence 0.210)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.682 [0.644, 0.719] | 0.682 [0.629, 0.730] | 0.675 [0.624, 0.722] |
| Confidence at the finding word | baseline | 0.614 [0.578, 0.647] | 0.601 [0.552, 0.651] | 0.600 [0.554, 0.646] |
| Self-consistency (3 sampled reports) | baseline | 0.669 [0.628, 0.706] | 0.640 [0.586, 0.691] | 0.647 [0.596, 0.697] |
| Self-verification P(yes), no training | baseline | 0.557 [0.517, 0.597] | 0.543 [0.487, 0.597] | 0.543 [0.487, 0.597] |
| External classifier (not trained on IU) | baseline | 0.669 [0.632, 0.704] | 0.675 [0.625, 0.723] | 0.648 [0.601, 0.696] |
| Image-level attention (control) | control | 0.687 [0.651, 0.725] | 0.675 [0.624, 0.721] | 0.669 [0.615, 0.720] |
| VERA-probe, offline region features | control-pair | 0.773 [0.736, 0.805] | 0.758 [0.714, 0.801] | 0.744 [0.698, 0.788] |
| VERA-probe, SHUFFLED regions (control) | control | 0.745 [0.707, 0.779] | 0.733 [0.686, 0.779] | 0.724 [0.673, 0.773] |
| VERA-probe (finding-word attention) | method | 0.774 [0.738, 0.807] | 0.759 [0.715, 0.801] | 0.745 [0.699, 0.789] |
| VERA-probe + sampled-report training | method | 0.809 [0.780, 0.839] | 0.798 [0.756, 0.836] | 0.762 [0.717, 0.805] |
| **VERA (full, model-internal)** | headline | 0.814 [0.783, 0.842] | 0.798 [0.757, 0.835] | 0.768 [0.726, 0.809] |
| VERA (full) + external classifier | method | 0.816 [0.785, 0.845] | 0.808 [0.766, 0.845] | 0.787 [0.741, 0.828] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.092 [+0.060, +0.122], p=0.001 | +0.070 [+0.036, +0.103], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.160 [+0.121, +0.199], p=0.001 | +0.145 [+0.098, +0.190], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.027 [+0.006, +0.048], p=0.005 | +0.021 [-0.014, +0.055], p=0.116 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.087 [+0.049, +0.124], p=0.001 | +0.076 [+0.018, +0.136], p=0.006 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.035 [+0.019, +0.053], p=0.001 | +0.017 [+0.000, +0.033], p=0.023 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.004 [-0.011, +0.017], p=0.273 | +0.007 [-0.014, +0.026], p=0.281 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.145 [+0.109, +0.183], p=0.001 | +0.121 [+0.078, +0.161], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.257 [+0.208, +0.300], p=0.001 | +0.226 [+0.152, +0.296], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.131 [+0.096, +0.164], p=0.001 | +0.093 [+0.060, +0.126], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | +0.002 [-0.013, +0.017], p=0.394 | +0.019 [-0.002, +0.042], p=0.035 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.145 [+0.103, +0.186], p=0.001 | +0.120 [+0.065, +0.176], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.610, recall 0.291; flag 20%: precision 0.542, recall 0.516; flag 30%: precision 0.462, recall 0.660; flag 50%: precision 0.363, recall 0.863

Headline per tier (5-fold CV): critical 0.658 [0.492, 0.809] (n=94); moderate 0.834 [0.805, 0.862] (n=1197); mild 0.546 [0.379, 0.724] (n=64)