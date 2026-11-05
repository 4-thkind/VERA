# VERA v6

- 686 images; 949 greedy primary claims; 4300 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 1360, 'A_s': 1360, 'T': 73, 'C': 12, 'S': 5, 'Q': 3, 'V': 21, 'I': 272, 'Aoff': 1360, 'Aoff_shuf': 1360}

## strict labels (n=723, prevalence 0.246)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.663 [0.611, 0.716] | 0.661 [0.595, 0.724] | 0.654 [0.588, 0.720] |
| Confidence at the finding word | baseline | 0.605 [0.559, 0.654] | 0.598 [0.538, 0.651] | 0.547 [0.480, 0.608] |
| Self-consistency (3 sampled reports) | baseline | 0.664 [0.612, 0.718] | 0.636 [0.567, 0.701] | 0.654 [0.594, 0.712] |
| Self-verification P(yes), no training | baseline | 0.660 [0.608, 0.711] | 0.649 [0.589, 0.708] | 0.649 [0.589, 0.708] |
| External classifier (not trained on IU) | baseline | 0.711 [0.660, 0.760] | 0.701 [0.640, 0.766] | 0.718 [0.658, 0.782] |
| Image-level attention (control) | control | 0.698 [0.649, 0.750] | 0.686 [0.628, 0.752] | 0.686 [0.631, 0.749] |
| VERA-probe, offline region features | control-pair | 0.720 [0.675, 0.764] | 0.726 [0.676, 0.777] | 0.742 [0.690, 0.793] |
| VERA-probe, SHUFFLED regions (control) | control | 0.701 [0.648, 0.752] | 0.694 [0.635, 0.754] | 0.689 [0.627, 0.754] |
| VERA-probe (finding-word attention) | method | 0.720 [0.675, 0.763] | 0.725 [0.674, 0.776] | 0.741 [0.689, 0.792] |
| VERA-probe + sampled-report training | method | 0.738 [0.696, 0.781] | 0.733 [0.682, 0.784] | 0.753 [0.703, 0.804] |
| **VERA (full, model-internal)** | headline | 0.745 [0.703, 0.788] | 0.722 [0.666, 0.781] | 0.726 [0.666, 0.782] |
| VERA (full) + external classifier | method | 0.762 [0.718, 0.806] | 0.739 [0.675, 0.801] | 0.748 [0.686, 0.808] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.058 [+0.015, +0.100], p=0.007 | +0.087 [+0.037, +0.135], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.115 [+0.069, +0.165], p=0.001 | +0.194 [+0.133, +0.261], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.019 [-0.016, +0.051], p=0.151 | +0.052 [+0.011, +0.097], p=0.010 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.023 [-0.023, +0.068], p=0.163 | +0.055 [-0.001, +0.104], p=0.028 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.018 [+0.003, +0.033], p=0.011 | +0.011 [-0.010, +0.034], p=0.171 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.007 [-0.023, +0.037], p=0.323 | -0.026 [-0.075, +0.018], p=0.872 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.081 [+0.043, +0.121], p=0.001 | +0.072 [+0.031, +0.113], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.084 [+0.033, +0.136], p=0.002 | +0.077 [+0.014, +0.140], p=0.007 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.082 [+0.037, +0.130], p=0.001 | +0.072 [+0.015, +0.127], p=0.004 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | +0.017 [-0.012, +0.043], p=0.118 | +0.022 [-0.012, +0.062], p=0.089 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.034 [-0.012, +0.082], p=0.077 | +0.008 [-0.053, +0.064], p=0.428 |

Headline operating points (5-fold CV): flag 10%: precision 0.630, recall 0.258; flag 20%: precision 0.497, recall 0.404; flag 30%: precision 0.470, recall 0.573; flag 50%: precision 0.378, recall 0.770

Headline per tier (5-fold CV): critical 0.840 [0.748, 0.926] (n=80); moderate 0.730 [0.681, 0.775] (n=609); mild 0.875 [0.689, 1.000] (n=34)

## broad labels (n=949, prevalence 0.426)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.629 [0.590, 0.666] | 0.630 [0.580, 0.679] | 0.636 [0.588, 0.683] |
| Confidence at the finding word | baseline | 0.602 [0.567, 0.637] | 0.602 [0.557, 0.647] | 0.596 [0.552, 0.641] |
| Self-consistency (3 sampled reports) | baseline | 0.667 [0.631, 0.703] | 0.659 [0.609, 0.707] | 0.661 [0.613, 0.706] |
| Self-verification P(yes), no training | baseline | 0.675 [0.639, 0.713] | 0.660 [0.617, 0.702] | 0.660 [0.617, 0.702] |
| External classifier (not trained on IU) | baseline | 0.676 [0.639, 0.715] | 0.664 [0.615, 0.711] | 0.657 [0.609, 0.705] |
| Image-level attention (control) | control | 0.671 [0.632, 0.707] | 0.662 [0.613, 0.708] | 0.645 [0.596, 0.695] |
| VERA-probe, offline region features | control-pair | 0.712 [0.677, 0.745] | 0.702 [0.659, 0.744] | 0.684 [0.640, 0.727] |
| VERA-probe, SHUFFLED regions (control) | control | 0.694 [0.658, 0.728] | 0.687 [0.639, 0.732] | 0.680 [0.635, 0.726] |
| VERA-probe (finding-word attention) | method | 0.715 [0.681, 0.748] | 0.706 [0.663, 0.747] | 0.684 [0.640, 0.726] |
| VERA-probe + sampled-report training | method | 0.721 [0.686, 0.754] | 0.718 [0.676, 0.758] | 0.711 [0.668, 0.753] |
| **VERA (full, model-internal)** | headline | 0.742 [0.710, 0.777] | 0.732 [0.684, 0.772] | 0.726 [0.678, 0.769] |
| VERA (full) + external classifier | method | 0.747 [0.713, 0.780] | 0.735 [0.690, 0.775] | 0.726 [0.680, 0.769] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.086 [+0.054, +0.118], p=0.001 | +0.048 [+0.007, +0.091], p=0.012 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.113 [+0.074, +0.151], p=0.001 | +0.087 [+0.038, +0.137], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.018 [-0.003, +0.039], p=0.050 | +0.004 [-0.021, +0.031], p=0.348 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.044 [+0.010, +0.076], p=0.003 | +0.039 [+0.000, +0.077], p=0.026 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.006 [-0.009, +0.021], p=0.221 | +0.027 [+0.008, +0.046], p=0.004 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.021 [+0.001, +0.041], p=0.022 | +0.015 [-0.014, +0.041], p=0.174 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.074 [+0.045, +0.103], p=0.001 | +0.065 [+0.030, +0.100], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.067 [+0.035, +0.097], p=0.001 | +0.066 [+0.023, +0.109], p=0.004 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.112 [+0.075, +0.150], p=0.001 | +0.090 [+0.041, +0.139], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | +0.005 [-0.010, +0.021], p=0.263 | -0.000 [-0.024, +0.024], p=0.512 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.065 [+0.026, +0.103], p=0.001 | +0.069 [+0.023, +0.117], p=0.004 |

Headline operating points (5-fold CV): flag 10%: precision 0.726, recall 0.171; flag 20%: precision 0.689, recall 0.324; flag 30%: precision 0.674, recall 0.475; flag 50%: precision 0.613, recall 0.720

Headline per tier (5-fold CV): critical 0.800 [0.711, 0.887] (n=95); moderate 0.729 [0.694, 0.768] (n=773); mild 0.766 [0.647, 0.864] (n=81)