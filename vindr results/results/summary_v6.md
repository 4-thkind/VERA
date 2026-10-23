# VERA v6

- 1670 images; 3569 greedy primary claims; 9954 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 5120, 'A_s': 5120, 'T': 64, 'C': 12, 'S': 5, 'Q': 3, 'V': 21, 'I': 1024, 'Aoff': 5120, 'Aoff_shuf': 5120}

## strict labels (n=3569, prevalence 0.116)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.608 [0.567, 0.646] | 0.623 [0.572, 0.668] | 0.575 [0.524, 0.622] |
| Confidence at the finding word | baseline | 0.669 [0.636, 0.699] | 0.653 [0.612, 0.695] | 0.665 [0.627, 0.702] |
| Self-consistency (3 sampled reports) | baseline | 0.709 [0.677, 0.739] | 0.696 [0.653, 0.737] | 0.699 [0.661, 0.739] |
| Self-verification P(yes), no training | baseline | 0.770 [0.743, 0.795] | 0.767 [0.730, 0.803] | 0.767 [0.730, 0.803] |
| External classifier (not trained on IU) | baseline | 0.588 [0.557, 0.620] | 0.569 [0.530, 0.608] | 0.586 [0.547, 0.623] |
| Image-level attention (control) | control | 0.765 [0.735, 0.791] | 0.779 [0.743, 0.813] | 0.765 [0.728, 0.799] |
| VERA-probe, offline region features | control-pair | 0.833 [0.805, 0.856] | 0.836 [0.801, 0.865] | 0.813 [0.782, 0.841] |
| VERA-probe, SHUFFLED regions (control) | control | 0.825 [0.796, 0.847] | 0.833 [0.799, 0.862] | 0.811 [0.782, 0.838] |
| VERA-probe (finding-word attention) | method | 0.837 [0.810, 0.859] | 0.841 [0.806, 0.870] | 0.807 [0.771, 0.840] |
| VERA-probe + sampled-report training | method | 0.860 [0.837, 0.880] | 0.861 [0.833, 0.888] | 0.834 [0.802, 0.863] |
| **VERA (full, model-internal)** | headline | 0.857 [0.836, 0.877] | 0.855 [0.827, 0.882] | 0.838 [0.809, 0.867] |
| VERA (full) + external classifier | method | 0.849 [0.828, 0.870] | 0.844 [0.815, 0.870] | 0.834 [0.804, 0.862] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.229 [+0.187, +0.268], p=0.001 | +0.232 [+0.187, +0.282], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.168 [+0.132, +0.199], p=0.001 | +0.143 [+0.102, +0.184], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.008 [-0.004, +0.020], p=0.089 | +0.002 [-0.016, +0.018], p=0.409 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.072 [+0.047, +0.096], p=0.001 | +0.043 [+0.009, +0.075], p=0.004 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.023 [+0.008, +0.040], p=0.002 | +0.027 [+0.010, +0.047], p=0.001 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | -0.002 [-0.015, +0.010], p=0.639 | +0.004 [-0.014, +0.021], p=0.337 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.149 [+0.124, +0.176], p=0.001 | +0.139 [+0.105, +0.170], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.088 [+0.064, +0.110], p=0.001 | +0.072 [+0.045, +0.102], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.249 [+0.209, +0.289], p=0.001 | +0.263 [+0.213, +0.315], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.008 [-0.013, -0.002], p=1.000 | -0.005 [-0.015, +0.007], p=0.801 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.270 [+0.232, +0.303], p=0.001 | +0.252 [+0.208, +0.296], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.501, recall 0.432; flag 20%: precision 0.382, recall 0.659; flag 30%: precision 0.307, recall 0.795; flag 50%: precision 0.215, recall 0.925

Headline per tier (5-fold CV): critical 0.858 [0.783, 0.923] (n=211); moderate 0.856 [0.829, 0.882] (n=1769); mild 0.859 [0.824, 0.888] (n=1589)

## broad labels (n=3569, prevalence 0.116)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.608 [0.567, 0.646] | 0.623 [0.572, 0.668] | 0.575 [0.524, 0.622] |
| Confidence at the finding word | baseline | 0.669 [0.636, 0.699] | 0.653 [0.612, 0.695] | 0.665 [0.627, 0.702] |
| Self-consistency (3 sampled reports) | baseline | 0.709 [0.677, 0.739] | 0.696 [0.653, 0.737] | 0.699 [0.661, 0.739] |
| Self-verification P(yes), no training | baseline | 0.770 [0.743, 0.795] | 0.767 [0.730, 0.803] | 0.767 [0.730, 0.803] |
| External classifier (not trained on IU) | baseline | 0.588 [0.557, 0.620] | 0.569 [0.530, 0.608] | 0.586 [0.547, 0.623] |
| Image-level attention (control) | control | 0.765 [0.735, 0.791] | 0.779 [0.743, 0.813] | 0.765 [0.728, 0.799] |
| VERA-probe, offline region features | control-pair | 0.833 [0.805, 0.856] | 0.836 [0.801, 0.865] | 0.813 [0.782, 0.841] |
| VERA-probe, SHUFFLED regions (control) | control | 0.825 [0.796, 0.847] | 0.833 [0.799, 0.862] | 0.811 [0.782, 0.838] |
| VERA-probe (finding-word attention) | method | 0.837 [0.810, 0.859] | 0.841 [0.806, 0.870] | 0.807 [0.771, 0.840] |
| VERA-probe + sampled-report training | method | 0.860 [0.837, 0.880] | 0.861 [0.833, 0.888] | 0.834 [0.802, 0.863] |
| **VERA (full, model-internal)** | headline | 0.857 [0.836, 0.877] | 0.855 [0.827, 0.882] | 0.838 [0.809, 0.867] |
| VERA (full) + external classifier | method | 0.849 [0.828, 0.870] | 0.844 [0.815, 0.870] | 0.834 [0.804, 0.862] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.229 [+0.187, +0.268], p=0.001 | +0.232 [+0.187, +0.282], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.168 [+0.132, +0.199], p=0.001 | +0.143 [+0.102, +0.184], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.008 [-0.004, +0.020], p=0.089 | +0.002 [-0.016, +0.018], p=0.409 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.072 [+0.047, +0.096], p=0.001 | +0.043 [+0.009, +0.075], p=0.004 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.023 [+0.008, +0.040], p=0.002 | +0.027 [+0.010, +0.047], p=0.001 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | -0.002 [-0.015, +0.010], p=0.639 | +0.004 [-0.014, +0.021], p=0.337 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.149 [+0.124, +0.176], p=0.001 | +0.139 [+0.105, +0.170], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.088 [+0.064, +0.110], p=0.001 | +0.072 [+0.045, +0.102], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.249 [+0.209, +0.289], p=0.001 | +0.263 [+0.213, +0.315], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.008 [-0.013, -0.002], p=1.000 | -0.005 [-0.015, +0.007], p=0.801 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.270 [+0.232, +0.303], p=0.001 | +0.252 [+0.208, +0.296], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.501, recall 0.432; flag 20%: precision 0.382, recall 0.659; flag 30%: precision 0.307, recall 0.795; flag 50%: precision 0.215, recall 0.925

Headline per tier (5-fold CV): critical 0.858 [0.783, 0.923] (n=211); moderate 0.856 [0.829, 0.882] (n=1769); mild 0.859 [0.824, 0.888] (n=1589)