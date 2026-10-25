# VERA v6

- 1061 images; 1321 greedy primary claims; 6226 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 5120, 'A_s': 5120, 'T': 113, 'C': 12, 'S': 5, 'Q': None, 'V': 21, 'I': 1024, 'Aoff': 5120, 'Aoff_shuf': 5120}

## strict labels (n=872, prevalence 0.236)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.650 [0.603, 0.696] | 0.635 [0.572, 0.692] | 0.595 [0.533, 0.653] |
| Confidence at the finding word | baseline | 0.640 [0.598, 0.684] | 0.639 [0.579, 0.691] | 0.671 [0.612, 0.719] |
| Self-consistency (3 sampled reports) | baseline | 0.725 [0.685, 0.763] | 0.748 [0.698, 0.796] | 0.749 [0.698, 0.797] |
| External classifier (not trained on IU) | baseline | 0.604 [0.558, 0.650] | 0.627 [0.566, 0.686] | 0.626 [0.563, 0.684] |
| Image-level attention (control) | control | 0.691 [0.643, 0.737] | 0.713 [0.660, 0.766] | 0.714 [0.661, 0.766] |
| VERA-probe, offline region features | control-pair | 0.725 [0.685, 0.765] | 0.722 [0.664, 0.772] | 0.705 [0.645, 0.759] |
| VERA-probe, SHUFFLED regions (control) | control | 0.710 [0.663, 0.752] | 0.718 [0.659, 0.772] | 0.697 [0.642, 0.751] |
| VERA-probe (finding-word attention) | method | 0.726 [0.684, 0.765] | 0.723 [0.666, 0.775] | 0.709 [0.651, 0.762] |
| VERA-probe + sampled-report training | method | 0.770 [0.731, 0.807] | 0.765 [0.717, 0.812] | 0.731 [0.681, 0.780] |
| **VERA (full, model-internal)** | headline | 0.790 [0.755, 0.825] | 0.793 [0.743, 0.839] | 0.796 [0.751, 0.840] |
| VERA (full) + external classifier | method | 0.785 [0.751, 0.820] | 0.791 [0.742, 0.835] | 0.793 [0.745, 0.837] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.075 [+0.028, +0.119], p=0.001 | +0.115 [+0.056, +0.174], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.085 [+0.039, +0.132], p=0.001 | +0.039 [-0.030, +0.099], p=0.121 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.015 [-0.014, +0.044], p=0.165 | +0.007 [-0.040, +0.055], p=0.412 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.035 [-0.000, +0.071], p=0.027 | -0.005 [-0.061, +0.047], p=0.569 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.044 [+0.021, +0.069], p=0.001 | +0.022 [-0.008, +0.054], p=0.081 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.020 [-0.006, +0.047], p=0.065 | +0.065 [+0.035, +0.101], p=0.001 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.065 [+0.036, +0.097], p=0.001 | +0.048 [+0.004, +0.088], p=0.015 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.140 [+0.087, +0.191], p=0.001 | +0.202 [+0.145, +0.265], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.005 [-0.020, +0.011], p=0.717 | -0.003 [-0.021, +0.013], p=0.654 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.186 [+0.130, +0.244], p=0.001 | +0.170 [+0.100, +0.244], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.670, recall 0.286; flag 20%: precision 0.600, recall 0.510; flag 30%: precision 0.492, recall 0.626; flag 50%: precision 0.385, recall 0.816

Headline per tier (5-fold CV): critical 0.852 [0.777, 0.917] (n=145); moderate 0.756 [0.704, 0.809] (n=460); mild 0.817 [0.756, 0.874] (n=267)

## broad labels (n=1321, prevalence 0.496)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.704 [0.677, 0.733] | 0.687 [0.650, 0.726] | 0.663 [0.628, 0.702] |
| Confidence at the finding word | baseline | 0.611 [0.582, 0.641] | 0.611 [0.575, 0.652] | 0.609 [0.572, 0.650] |
| Self-consistency (3 sampled reports) | baseline | 0.716 [0.686, 0.745] | 0.727 [0.691, 0.762] | 0.733 [0.698, 0.768] |
| External classifier (not trained on IU) | baseline | 0.658 [0.626, 0.689] | 0.663 [0.622, 0.705] | 0.653 [0.613, 0.694] |
| Image-level attention (control) | control | 0.703 [0.675, 0.734] | 0.708 [0.670, 0.744] | 0.680 [0.642, 0.715] |
| VERA-probe, offline region features | control-pair | 0.757 [0.732, 0.785] | 0.749 [0.714, 0.783] | 0.722 [0.684, 0.759] |
| VERA-probe, SHUFFLED regions (control) | control | 0.735 [0.709, 0.764] | 0.722 [0.689, 0.758] | 0.713 [0.680, 0.748] |
| VERA-probe (finding-word attention) | method | 0.757 [0.731, 0.784] | 0.747 [0.713, 0.782] | 0.726 [0.689, 0.764] |
| VERA-probe + sampled-report training | method | 0.774 [0.749, 0.801] | 0.767 [0.733, 0.799] | 0.753 [0.719, 0.787] |
| **VERA (full, model-internal)** | headline | 0.790 [0.765, 0.815] | 0.789 [0.757, 0.821] | 0.781 [0.750, 0.812] |
| VERA (full) + external classifier | method | 0.785 [0.760, 0.809] | 0.786 [0.755, 0.818] | 0.785 [0.755, 0.816] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.053 [+0.025, +0.083], p=0.001 | +0.063 [+0.026, +0.097], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.146 [+0.113, +0.178], p=0.001 | +0.118 [+0.067, +0.161], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.022 [+0.005, +0.038], p=0.006 | +0.009 [-0.017, +0.035], p=0.243 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.054 [+0.027, +0.079], p=0.001 | +0.047 [+0.011, +0.083], p=0.007 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.018 [+0.006, +0.030], p=0.004 | +0.026 [+0.007, +0.045], p=0.006 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.016 [+0.002, +0.028], p=0.010 | +0.028 [+0.009, +0.050], p=0.004 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.074 [+0.055, +0.096], p=0.001 | +0.048 [+0.023, +0.073], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.087 [+0.059, +0.117], p=0.001 | +0.118 [+0.079, +0.151], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.006 [-0.015, +0.005], p=0.878 | +0.005 [-0.008, +0.018], p=0.221 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.132 [+0.097, +0.168], p=0.001 | +0.128 [+0.079, +0.175], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.902, recall 0.183; flag 20%: precision 0.864, recall 0.350; flag 30%: precision 0.811, recall 0.492; flag 50%: precision 0.700, recall 0.707

Headline per tier (5-fold CV): critical 0.775 [0.699, 0.849] (n=185); moderate 0.770 [0.736, 0.803] (n=656); mild 0.812 [0.773, 0.851] (n=480)