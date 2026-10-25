# VERA v6

- 870 images; 1087 greedy primary claims; 5630 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 5120, 'A_s': 5120, 'T': 115, 'C': 12, 'S': 5, 'Q': None, 'V': 21, 'I': 1024, 'Aoff': 5120, 'Aoff_shuf': 5120}

## strict labels (n=1087, prevalence 0.262)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.796 [0.766, 0.826] | 0.792 [0.753, 0.831] | 0.814 [0.773, 0.852] |
| Confidence at the finding word | baseline | 0.644 [0.607, 0.680] | 0.647 [0.597, 0.694] | 0.661 [0.612, 0.708] |
| Self-consistency (3 sampled reports) | baseline | 0.721 [0.690, 0.753] | 0.747 [0.705, 0.786] | 0.748 [0.706, 0.786] |
| External classifier (not trained on IU) | baseline | 0.704 [0.666, 0.737] | 0.698 [0.649, 0.744] | 0.673 [0.624, 0.720] |
| Image-level attention (control) | control | 0.782 [0.749, 0.814] | 0.780 [0.733, 0.823] | 0.776 [0.733, 0.820] |
| VERA-probe, offline region features | control-pair | 0.842 [0.814, 0.869] | 0.838 [0.801, 0.874] | 0.826 [0.790, 0.863] |
| VERA-probe, SHUFFLED regions (control) | control | 0.826 [0.797, 0.854] | 0.825 [0.788, 0.863] | 0.818 [0.775, 0.857] |
| VERA-probe (finding-word attention) | method | 0.840 [0.810, 0.867] | 0.834 [0.796, 0.870] | 0.823 [0.784, 0.859] |
| VERA-probe + sampled-report training | method | 0.860 [0.835, 0.883] | 0.845 [0.810, 0.877] | 0.840 [0.804, 0.873] |
| **VERA (full, model-internal)** | headline | 0.865 [0.842, 0.888] | 0.860 [0.829, 0.889] | 0.863 [0.830, 0.893] |
| VERA (full) + external classifier | method | 0.861 [0.836, 0.884] | 0.856 [0.824, 0.885] | 0.846 [0.812, 0.881] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.044 [+0.021, +0.071], p=0.001 | +0.009 [-0.024, +0.045], p=0.303 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.196 [+0.158, +0.233], p=0.001 | +0.161 [+0.112, +0.219], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.016 [-0.000, +0.034], p=0.031 | +0.008 [-0.016, +0.035], p=0.250 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.058 [+0.031, +0.086], p=0.001 | +0.047 [+0.011, +0.081], p=0.005 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.019 [+0.004, +0.036], p=0.014 | +0.017 [-0.006, +0.039], p=0.090 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.006 [-0.005, +0.016], p=0.118 | +0.023 [+0.010, +0.037], p=0.001 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.144 [+0.115, +0.175], p=0.001 | +0.115 [+0.078, +0.154], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.070 [+0.046, +0.095], p=0.001 | +0.049 [+0.019, +0.078], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.005 [-0.012, +0.001], p=0.936 | -0.017 [-0.030, -0.002], p=0.990 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.162 [+0.127, +0.201], p=0.001 | +0.189 [+0.138, +0.240], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.826, recall 0.316; flag 20%: precision 0.716, recall 0.547; flag 30%: precision 0.604, recall 0.691; flag 50%: precision 0.474, recall 0.905

Headline per tier (5-fold CV): critical 0.904 [0.856, 0.943] (n=218); moderate 0.884 [0.853, 0.911] (n=661); mild 0.752 [0.673, 0.823] (n=208)

## broad labels (n=1087, prevalence 0.262)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.796 [0.766, 0.826] | 0.792 [0.753, 0.831] | 0.814 [0.773, 0.852] |
| Confidence at the finding word | baseline | 0.644 [0.607, 0.680] | 0.647 [0.597, 0.694] | 0.661 [0.612, 0.708] |
| Self-consistency (3 sampled reports) | baseline | 0.721 [0.690, 0.753] | 0.747 [0.705, 0.786] | 0.748 [0.706, 0.786] |
| External classifier (not trained on IU) | baseline | 0.704 [0.666, 0.737] | 0.698 [0.649, 0.744] | 0.673 [0.624, 0.720] |
| Image-level attention (control) | control | 0.782 [0.749, 0.814] | 0.780 [0.733, 0.823] | 0.776 [0.733, 0.820] |
| VERA-probe, offline region features | control-pair | 0.842 [0.814, 0.869] | 0.838 [0.801, 0.874] | 0.826 [0.790, 0.863] |
| VERA-probe, SHUFFLED regions (control) | control | 0.826 [0.797, 0.854] | 0.825 [0.788, 0.863] | 0.818 [0.775, 0.857] |
| VERA-probe (finding-word attention) | method | 0.840 [0.810, 0.867] | 0.834 [0.796, 0.870] | 0.823 [0.784, 0.859] |
| VERA-probe + sampled-report training | method | 0.860 [0.835, 0.883] | 0.845 [0.810, 0.877] | 0.840 [0.804, 0.873] |
| **VERA (full, model-internal)** | headline | 0.865 [0.842, 0.888] | 0.860 [0.829, 0.889] | 0.863 [0.830, 0.893] |
| VERA (full) + external classifier | method | 0.861 [0.836, 0.884] | 0.856 [0.824, 0.885] | 0.846 [0.812, 0.881] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.044 [+0.021, +0.071], p=0.001 | +0.009 [-0.024, +0.045], p=0.303 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.196 [+0.158, +0.233], p=0.001 | +0.161 [+0.112, +0.219], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.016 [-0.000, +0.034], p=0.031 | +0.008 [-0.016, +0.035], p=0.250 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.058 [+0.031, +0.086], p=0.001 | +0.047 [+0.011, +0.081], p=0.005 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.019 [+0.004, +0.036], p=0.014 | +0.017 [-0.006, +0.039], p=0.090 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.006 [-0.005, +0.016], p=0.118 | +0.023 [+0.010, +0.037], p=0.001 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.144 [+0.115, +0.175], p=0.001 | +0.115 [+0.078, +0.154], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.070 [+0.046, +0.095], p=0.001 | +0.049 [+0.019, +0.078], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.005 [-0.012, +0.001], p=0.936 | -0.017 [-0.030, -0.002], p=0.990 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.162 [+0.127, +0.201], p=0.001 | +0.189 [+0.138, +0.240], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.826, recall 0.316; flag 20%: precision 0.716, recall 0.547; flag 30%: precision 0.604, recall 0.691; flag 50%: precision 0.474, recall 0.905

Headline per tier (5-fold CV): critical 0.904 [0.856, 0.943] (n=218); moderate 0.884 [0.853, 0.911] (n=661); mild 0.752 [0.673, 0.823] (n=208)