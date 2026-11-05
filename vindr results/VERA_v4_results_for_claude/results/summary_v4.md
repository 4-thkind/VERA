# VERA v4 detector

- Primary claims 2175 (calib 871, test 1304 from 765 images); test prevalence 0.586; clean-label test claims 701 (prevalence 0.230)
- **Selected (calibration only): Ensemble of top-3 by calibration CV**

## All candidates (AUROC, 95% CI)

| model | calib CV | test (held-out) | test, clean labels | cross-fit, test claims | cross-fit, all claims |
|---|---|---|---|---|---|
| Text only | 0.716 | 0.704 [0.672, 0.734] | 0.648 [0.597, 0.695] | 0.700 [0.667, 0.731] | 0.712 [0.688, 0.736] |
| Confidence only | 0.609 | 0.584 [0.549, 0.621] | 0.522 [0.466, 0.578] | 0.571 [0.536, 0.609] | 0.584 [0.557, 0.611] |
| Global attention, no region (G) | 0.752 | 0.738 [0.708, 0.770] | 0.736 [0.684, 0.784] | 0.764 [0.736, 0.794] | 0.768 [0.747, 0.790] |
| Region alignment (R, VERA-probe) | 0.726 | 0.718 [0.691, 0.746] | 0.722 [0.674, 0.767] | 0.726 [0.698, 0.758] | 0.738 [0.717, 0.758] |
| Region alignment, SHUFFLED regions (control) | 0.655 | 0.607 [0.576, 0.639] | 0.626 [0.578, 0.679] | 0.667 [0.636, 0.698] | 0.667 [0.644, 0.689] |
| Image-level attention only (control) | 0.695 | 0.652 [0.621, 0.688] | 0.655 [0.601, 0.706] | 0.671 [0.639, 0.705] | 0.690 [0.666, 0.714] |
| Global + region (attention only) | 0.743 | 0.737 [0.709, 0.768] | 0.731 [0.679, 0.780] | 0.753 [0.726, 0.782] | 0.761 [0.740, 0.781] |
| Text + confidence | 0.717 | 0.716 [0.684, 0.745] | 0.660 [0.610, 0.706] | 0.705 [0.670, 0.738] | 0.712 [0.688, 0.737] |
| Text + confidence + global | 0.758 | 0.745 [0.715, 0.778] | 0.736 [0.685, 0.786] | 0.769 [0.741, 0.799] | 0.776 [0.754, 0.796] |
| Text + confidence + region | 0.741 | 0.729 [0.702, 0.756] | 0.710 [0.659, 0.755] | 0.739 [0.712, 0.771] | 0.753 [0.733, 0.774] |
| Full: text + confidence + global + region | 0.747 | 0.740 [0.711, 0.770] | 0.732 [0.680, 0.779] | 0.756 [0.728, 0.786] | 0.765 [0.745, 0.785] |
| Full with SHUFFLED regions (control) | 0.751 | 0.702 [0.672, 0.734] | 0.680 [0.629, 0.729] | 0.744 [0.715, 0.772] | 0.748 [0.726, 0.770] |
| Full, gradient boosting (per-layer summaries) | 0.727 | 0.742 [0.712, 0.772] | 0.725 [0.679, 0.771] | 0.727 [0.698, 0.756] | 0.738 [0.716, 0.760] |
| Ensemble of top-3 by calibration CV * | 0.760 | 0.745 [0.714, 0.777] | 0.738 [0.686, 0.787] | 0.768 [0.740, 0.797] | 0.774 [0.753, 0.795] |

## Paired comparisons (AUROC difference, image bootstrap)

| question | A vs B | test | test, clean | cross-fit test |
|---|---|---|---|---|
| attention adds over text + confidence | Full: text + confidence + global + region vs Text + confidence | +0.025 [-0.006, +0.059], p=0.053 | +0.072 [+0.020, +0.125], p=0.003 | +0.051 [+0.024, +0.082], p=0.001 |
| full detector vs text-only prior | Full: text + confidence + global + region vs Text only | +0.036 [+0.005, +0.070], p=0.011 | +0.084 [+0.036, +0.135], p=0.002 | +0.057 [+0.028, +0.086], p=0.001 |
| named region matters inside full detector | Full: text + confidence + global + region vs Full with SHUFFLED regions (control) | +0.038 [+0.019, +0.056], p=0.001 | +0.052 [+0.026, +0.081], p=0.001 | +0.013 [-0.005, +0.029], p=0.080 |
| named region matters (attention only) | Region alignment (R, VERA-probe) vs Region alignment, SHUFFLED regions (control) | +0.111 [+0.079, +0.143], p=0.001 | +0.096 [+0.046, +0.143], p=0.001 | +0.059 [+0.032, +0.085], p=0.001 |
| region adds over global attention | Global + region (attention only) vs Global attention, no region (G) | -0.001 [-0.013, +0.011], p=0.566 | -0.004 [-0.022, +0.014], p=0.669 | -0.011 [-0.025, +0.001], p=0.966 |
| region adds over text + confidence | Text + confidence + region vs Text + confidence | +0.014 [-0.013, +0.045], p=0.164 | +0.050 [+0.003, +0.102], p=0.017 | +0.035 [+0.009, +0.063], p=0.008 |
| claim-level beats image-level attention | Global attention, no region (G) vs Image-level attention only (control) | +0.086 [+0.060, +0.114], p=0.001 | +0.080 [+0.032, +0.127], p=0.002 | +0.093 [+0.067, +0.120], p=0.001 |
| selected detector vs text-only prior | Ensemble of top-3 by calibration CV vs Text only | +0.041 [+0.010, +0.075], p=0.007 | +0.090 [+0.040, +0.141], p=0.001 | +0.068 [+0.040, +0.098], p=0.001 |
| selected detector vs text + confidence | Ensemble of top-3 by calibration CV vs Text + confidence | +0.029 [-0.001, +0.063], p=0.029 | +0.078 [+0.025, +0.131], p=0.003 | +0.063 [+0.034, +0.094], p=0.001 |
| selected vs training-free | Ensemble of top-3 by calibration CV vs training-free VERA (v3 primary) | +0.146 [+0.112, +0.185], p=0.001 | +0.142 [+0.075, +0.209], p=0.001 | +0.169 [+0.134, +0.207], p=0.001 |

## Selected model per tier (test)
- critical: 0.792 [0.733, 0.852] (n=278); clean labels 0.772 [0.688, 0.854] (n=179)
- moderate: 0.727 [0.686, 0.770] (n=748); clean labels 0.756 [0.695, 0.813] (n=373)
- mild: 0.757 [0.695, 0.817] (n=278); clean labels 0.651 [0.473, 0.829] (n=149)

## Selected model operating points (thresholds from calibration)
- flag 10%: test flag rate 12.4%, precision 0.772, recall 0.164, F1 0.270
- flag 20%: test flag rate 22.2%, precision 0.817, recall 0.309, F1 0.448
- flag 30%: test flag rate 32.0%, precision 0.808, recall 0.441, F1 0.571
- flag 50%: test flag rate 48.5%, precision 0.773, recall 0.640, F1 0.700

Full linear model, share of |coefficients| per family: {'T': 0.02387915551662445, 'C': 0.0042636157013475895, 'G': 0.3839738368988037, 'R': 0.5878834128379822}