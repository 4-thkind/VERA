# VERA v6

- 1292 images; 1993 greedy primary claims; 5032 claims from sampled reports (training only)
- Headline (pre-registered): **VERA (full, model-internal)**; primary protocol: 5-fold CV grouped by image; primary labels: strict
- Feature families: {'A': 5120, 'A_s': 5120, 'T': 101, 'C': 12, 'S': 5, 'Q': 3, 'V': 21, 'I': 1024, 'Aoff': 5120, 'Aoff_shuf': 5120}

## strict labels (n=1359, prevalence 0.333)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.660 [0.614, 0.706] | 0.694 [0.652, 0.731] | 0.652 [0.608, 0.694] |
| Confidence at the finding word | baseline | 0.570 [0.531, 0.611] | 0.602 [0.553, 0.648] | 0.602 [0.554, 0.647] |
| Self-consistency (3 sampled reports) | baseline | 0.693 [0.660, 0.726] | 0.703 [0.666, 0.746] | 0.700 [0.663, 0.741] |
| Self-verification P(yes), no training | baseline | 0.714 [0.683, 0.743] | 0.713 [0.675, 0.750] | 0.713 [0.675, 0.750] |
| External classifier (not trained on IU) | baseline | 0.616 [0.580, 0.652] | 0.626 [0.581, 0.668] | 0.620 [0.571, 0.666] |
| Image-level attention (control) | control | 0.724 [0.693, 0.756] | 0.722 [0.677, 0.764] | 0.696 [0.652, 0.738] |
| VERA-probe, offline region features | control-pair | 0.806 [0.777, 0.831] | 0.798 [0.760, 0.832] | 0.759 [0.722, 0.796] |
| VERA-probe, SHUFFLED regions (control) | control | 0.796 [0.769, 0.821] | 0.791 [0.756, 0.825] | 0.747 [0.710, 0.785] |
| VERA-probe (finding-word attention) | method | 0.802 [0.773, 0.827] | 0.794 [0.754, 0.829] | 0.758 [0.721, 0.793] |
| VERA-probe + sampled-report training | method | 0.817 [0.789, 0.843] | 0.809 [0.776, 0.840] | 0.788 [0.752, 0.821] |
| **VERA (full, model-internal)** | headline | 0.820 [0.793, 0.843] | 0.819 [0.787, 0.850] | 0.816 [0.783, 0.846] |
| VERA (full) + external classifier | method | 0.812 [0.784, 0.836] | 0.811 [0.778, 0.845] | 0.808 [0.774, 0.842] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.142 [+0.097, +0.191], p=0.001 | +0.106 [+0.065, +0.150], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.232 [+0.184, +0.279], p=0.001 | +0.156 [+0.103, +0.211], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | +0.010 [-0.005, +0.024], p=0.093 | +0.013 [-0.006, +0.032], p=0.097 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.078 [+0.050, +0.103], p=0.001 | +0.062 [+0.027, +0.100], p=0.001 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.015 [+0.003, +0.030], p=0.006 | +0.030 [+0.010, +0.049], p=0.002 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.003 [-0.014, +0.020], p=0.388 | +0.028 [+0.009, +0.047], p=0.003 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.127 [+0.100, +0.152], p=0.001 | +0.116 [+0.084, +0.147], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.106 [+0.077, +0.131], p=0.001 | +0.103 [+0.070, +0.135], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.159 [+0.115, +0.207], p=0.001 | +0.164 [+0.121, +0.207], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.007 [-0.018, +0.004], p=0.911 | -0.007 [-0.022, +0.006], p=0.849 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.204 [+0.167, +0.240], p=0.001 | +0.195 [+0.144, +0.242], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.772, recall 0.232; flag 20%: precision 0.735, recall 0.442; flag 30%: precision 0.664, recall 0.600; flag 50%: precision 0.546, recall 0.821

Headline per tier (5-fold CV): critical 0.852 [0.800, 0.898] (n=270); moderate 0.797 [0.766, 0.829] (n=842); mild 0.795 [0.712, 0.867] (n=247)

## broad labels (n=1993, prevalence 0.545)

| model | role | 5-fold CV (all images) | 5-fold CV, test-split images | held-out (calib -> test) |
|---|---|---|---|---|
| Text only | baseline | 0.674 [0.648, 0.701] | 0.674 [0.643, 0.703] | 0.642 [0.609, 0.675] |
| Confidence at the finding word | baseline | 0.588 [0.555, 0.618] | 0.602 [0.567, 0.636] | 0.577 [0.542, 0.612] |
| Self-consistency (3 sampled reports) | baseline | 0.673 [0.648, 0.698] | 0.672 [0.641, 0.704] | 0.676 [0.645, 0.707] |
| Self-verification P(yes), no training | baseline | 0.732 [0.709, 0.754] | 0.732 [0.704, 0.759] | 0.732 [0.704, 0.759] |
| External classifier (not trained on IU) | baseline | 0.626 [0.597, 0.655] | 0.628 [0.592, 0.664] | 0.633 [0.596, 0.669] |
| Image-level attention (control) | control | 0.721 [0.695, 0.745] | 0.712 [0.682, 0.742] | 0.695 [0.663, 0.724] |
| VERA-probe, offline region features | control-pair | 0.776 [0.753, 0.798] | 0.771 [0.742, 0.797] | 0.737 [0.707, 0.765] |
| VERA-probe, SHUFFLED regions (control) | control | 0.777 [0.755, 0.799] | 0.773 [0.747, 0.797] | 0.730 [0.701, 0.756] |
| VERA-probe (finding-word attention) | method | 0.773 [0.750, 0.795] | 0.771 [0.744, 0.795] | 0.733 [0.703, 0.762] |
| VERA-probe + sampled-report training | method | 0.786 [0.764, 0.808] | 0.781 [0.754, 0.806] | 0.762 [0.734, 0.787] |
| **VERA (full, model-internal)** | headline | 0.799 [0.780, 0.818] | 0.795 [0.771, 0.818] | 0.787 [0.763, 0.810] |
| VERA (full) + external classifier | method | 0.799 [0.778, 0.819] | 0.791 [0.765, 0.815] | 0.792 [0.766, 0.817] |

| question | A vs B | 5-fold CV | held-out |
|---|---|---|---|
| attention beats the text prior | VERA-probe (finding-word attention) vs Text only | +0.099 [+0.073, +0.127], p=0.001 | +0.091 [+0.052, +0.127], p=0.001 |
| attention beats token confidence | VERA-probe (finding-word attention) vs Confidence at the finding word | +0.185 [+0.148, +0.223], p=0.001 | +0.156 [+0.118, +0.195], p=0.001 |
| the NAMED region matters (grounding) | VERA-probe, offline region features vs VERA-probe, SHUFFLED regions (control) | -0.002 [-0.013, +0.011], p=0.583 | +0.007 [-0.010, +0.024], p=0.211 |
| claim-level beats image-level attention | VERA-probe (finding-word attention) vs Image-level attention (control) | +0.052 [+0.032, +0.074], p=0.001 | +0.038 [+0.011, +0.066], p=0.003 |
| sampled reports as extra training data help | VERA-probe + sampled-report training vs VERA-probe (finding-word attention) | +0.013 [+0.001, +0.025], p=0.011 | +0.029 [+0.011, +0.046], p=0.001 |
| consistency + self-verification + confidence add | VERA (full, model-internal) vs VERA-probe + sampled-report training | +0.013 [+0.001, +0.024], p=0.019 | +0.025 [+0.006, +0.043], p=0.010 |
| full vs self-consistency alone | VERA (full, model-internal) vs Self-consistency (3 sampled reports) | +0.126 [+0.104, +0.149], p=0.001 | +0.111 [+0.085, +0.136], p=0.001 |
| full vs self-verification alone | VERA (full, model-internal) vs Self-verification P(yes), no training | +0.067 [+0.049, +0.084], p=0.001 | +0.055 [+0.033, +0.077], p=0.001 |
| full vs text prior | VERA (full, model-internal) vs Text only | +0.125 [+0.099, +0.153], p=0.001 | +0.145 [+0.108, +0.180], p=0.001 |
| an external classifier adds | VERA (full) + external classifier vs VERA (full, model-internal) | -0.000 [-0.009, +0.007], p=0.550 | +0.005 [-0.004, +0.014], p=0.138 |
| full vs external classifier | VERA (full, model-internal) vs External classifier (not trained on IU) | +0.173 [+0.143, +0.205], p=0.001 | +0.154 [+0.117, +0.194], p=0.001 |

Headline operating points (5-fold CV): flag 10%: precision 0.935, recall 0.172; flag 20%: precision 0.880, recall 0.323; flag 30%: precision 0.853, recall 0.470; flag 50%: precision 0.763, recall 0.701

Headline per tier (5-fold CV): critical 0.862 [0.822, 0.899] (n=346); moderate 0.783 [0.755, 0.808] (n=1180); mild 0.789 [0.747, 0.829] (n=467)