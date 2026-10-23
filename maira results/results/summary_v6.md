# VERA v6

- 987 images; 1232 greedy primary claims; 2276 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 5120, 'A_s': 5120, 'T': 113, 'C': 12, 'S': 5, 'Q': None, 'V': 21, 'I': 1024, 'Aoff': 5120, 'Aoff_shuf': 5120}

## strict labels (n=798, prevalence 0.193)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.690 [0.640, 0.737] | 0.648 [0.581, 0.713] | 0.593 [0.527, 0.659] |
| Confidence at the finding word | baseline | 0.656 [0.606, 0.699] | 0.620 [0.551, 0.686] | 0.620 [0.553, 0.685] |
| Self-consistency (3 sampled reports) | baseline | 0.706 [0.661, 0.747] | 0.706 [0.649, 0.767] | 0.707 [0.650, 0.765] |
| External classifier (not trained on IU) | baseline | 0.585 [0.535, 0.637] | 0.583 [0.515, 0.645] | 0.584 [0.517, 0.648] |
| Image-level attention (control) | control | 0.667 [0.616, 0.716] | 0.663 [0.595, 0.727] | 0.692 [0.626, 0.755] |
| VERA-probe, offline region features | control-pair | 0.710 [0.667, 0.753] | 0.677 [0.611, 0.739] | 0.664 [0.602, 0.721] |
| VERA-probe, SHUFFLED regions (control) | control | 0.713 [0.668, 0.760] | 0.691 [0.631, 0.749] | 0.676 [0.614, 0.736] |
| VERA-probe (finding-word attention) | method | 0.709 [0.664, 0.754] | 0.678 [0.616, 0.738] | 0.660 [0.598, 0.719] |
| VERA-probe + sampled-report training | method | 0.739 [0.697, 0.778] | 0.721 [0.663, 0.782] | 0.706 [0.650, 0.769] |
| **VERA (full, model-internal)** | headline | 0.772 [0.733, 0.806] | 0.750 [0.696, 0.805] | 0.737 [0.680, 0.797] |
| VERA (full) + external classifier | method | 0.775 [0.735, 0.808] | 0.754 [0.700, 0.803] | 0.741 [0.683, 0.798] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.019 [-0.031, +0.071], p=0.250 | +0.068 [+0.010, +0.126], p=0.013 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.053 [+0.001, +0.105], p=0.023 | +0.040 [-0.032, +0.113], p=0.132 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | -0.003 [-0.036, +0.025], p=0.610 | -0.013 [-0.056, +0.032], p=0.712 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.042 [+0.001, +0.081], p=0.020 | -0.032 [-0.090, +0.027], p=0.848 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.030 [+0.007, +0.054], p=0.010 | +0.046 [+0.013, +0.078], p=0.004 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.033 [+0.000, +0.065], p=0.025 | +0.031 [-0.021, +0.082], p=0.098 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.066 [+0.034, +0.097], p=0.001 | +0.030 [-0.018, +0.085], p=0.128 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.082 [+0.029, +0.136], p=0.001 | +0.144 [+0.075, +0.214], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | +0.002 [-0.015, +0.020], p=0.429 | +0.004 [-0.030, +0.031], p=0.411 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.187 [+0.121, +0.249], p=0.001 | +0.153 [+0.066, +0.250], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.562, recall 0.292; flag 20%: precision 0.438, recall 0.455; flag 30%: precision 0.375, recall 0.584; flag 50%: precision 0.318, recall 0.825

Headline per tier (5-fold CV): critical 0.821 [0.745, 0.890] (n=137); moderate 0.754 [0.700, 0.805] (n=424); mild 0.718 [0.603, 0.822] (n=237)

## broad labels (n=1232, prevalence 0.477)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.710 [0.681, 0.740] | 0.696 [0.660, 0.734] | 0.697 [0.658, 0.736] |
| Confidence at the finding word | baseline | 0.610 [0.578, 0.642] | 0.610 [0.570, 0.652] | 0.607 [0.568, 0.651] |
| Self-consistency (3 sampled reports) | baseline | 0.678 [0.649, 0.709] | 0.683 [0.643, 0.719] | 0.691 [0.654, 0.729] |
| External classifier (not trained on IU) | baseline | 0.665 [0.632, 0.696] | 0.688 [0.647, 0.722] | 0.656 [0.617, 0.691] |
| Image-level attention (control) | control | 0.706 [0.677, 0.737] | 0.714 [0.675, 0.754] | 0.716 [0.682, 0.753] |
| VERA-probe, offline region features | control-pair | 0.742 [0.715, 0.769] | 0.734 [0.699, 0.770] | 0.740 [0.707, 0.777] |
| VERA-probe, SHUFFLED regions (control) | control | 0.749 [0.722, 0.774] | 0.750 [0.718, 0.783] | 0.746 [0.712, 0.780] |
| VERA-probe (finding-word attention) | method | 0.740 [0.712, 0.767] | 0.730 [0.695, 0.766] | 0.745 [0.713, 0.781] |
| VERA-probe + sampled-report training | method | 0.759 [0.733, 0.785] | 0.752 [0.719, 0.789] | 0.757 [0.723, 0.794] |
| **VERA (full, model-internal)** | headline | 0.780 [0.752, 0.805] | 0.773 [0.742, 0.807] | 0.777 [0.744, 0.810] |
| VERA (full) + external classifier | method | 0.784 [0.758, 0.808] | 0.784 [0.754, 0.816] | 0.780 [0.747, 0.814] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.030 [+0.002, +0.055], p=0.019 | +0.048 [+0.015, +0.087], p=0.003 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.130 [+0.093, +0.165], p=0.001 | +0.138 [+0.093, +0.188], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | -0.007 [-0.023, +0.010], p=0.781 | -0.006 [-0.026, +0.016], p=0.714 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.034 [+0.006, +0.059], p=0.007 | +0.029 [-0.005, +0.063], p=0.043 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.020 [+0.008, +0.032], p=0.002 | +0.012 [-0.005, +0.029], p=0.083 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.021 [+0.006, +0.035], p=0.001 | +0.020 [-0.001, +0.040], p=0.035 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.102 [+0.076, +0.127], p=0.001 | +0.085 [+0.057, +0.118], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.070 [+0.039, +0.098], p=0.001 | +0.080 [+0.040, +0.120], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | +0.004 [-0.007, +0.014], p=0.197 | +0.003 [-0.010, +0.016], p=0.338 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.115 [+0.079, +0.154], p=0.001 | +0.121 [+0.078, +0.167], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.855, recall 0.180; flag 20%: precision 0.842, recall 0.354; flag 30%: precision 0.800, recall 0.503; flag 50%: precision 0.693, recall 0.726

Headline per tier (5-fold CV): critical 0.756 [0.677, 0.832] (n=175); moderate 0.761 [0.725, 0.800] (n=611); mild 0.804 [0.764, 0.842] (n=446)