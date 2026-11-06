# VERA v5 detector

- Primary claims 2352 (calib 960, test 1392 from 837 images); test prevalence 0.595; clean-label test claims 786 (prevalence 0.282)
- **Selected, VLM-internal only (calibration only): Ensemble (internal only): top-3 by calibration CV**
- **Selected, may use external verifier: Ensemble (internal only): top-3 by calibration CV**
- Feature families: {'T': 106, 'C': 11, 'G': 2144, 'I': 1025, 'R': 4288, 'Rs': 4288, 'compact': 405, 'U': 10, 'Xn': 5, 'Xc': 5, 'Xb': 6, 'X': 16, 'Gm': 2113, 'Rmn': 4160, 'Rmn_s': 4160, 'V': 21, 'compact5': 560}

## All candidates (AUROC, 95% CI)

| model | group | calib CV | test (held-out) | test, clean labels | cross-fit, test claims | cross-fit, all claims |
|---|---|---|---|---|---|---|
| Text only | internal | 0.688 | 0.696 [0.667, 0.725] | 0.622 [0.577, 0.665] | 0.712 [0.683, 0.739] | 0.708 [0.685, 0.732] |
| Confidence only | internal | 0.591 | 0.582 [0.551, 0.614] | 0.521 [0.468, 0.575] | 0.587 [0.556, 0.622] | 0.590 [0.565, 0.617] |
| Global attention, no region (G) | internal | 0.733 | 0.740 [0.711, 0.769] | 0.745 [0.696, 0.792] | 0.778 [0.752, 0.804] | 0.774 [0.753, 0.797] |
| Region alignment (R, VERA-probe) | internal | 0.726 | 0.720 [0.691, 0.749] | 0.738 [0.699, 0.780] | 0.742 [0.715, 0.770] | 0.745 [0.723, 0.766] |
| Region alignment, SHUFFLED regions (control) | internal, control | 0.625 | 0.645 [0.616, 0.674] | 0.649 [0.607, 0.689] | 0.677 [0.649, 0.706] | 0.668 [0.644, 0.692] |
| Image-level attention only (control) | internal, control | 0.683 | 0.683 [0.654, 0.712] | 0.722 [0.678, 0.763] | 0.693 [0.664, 0.721] | 0.700 [0.675, 0.727] |
| Global + region (attention only) | internal | 0.739 | 0.728 [0.699, 0.757] | 0.741 [0.690, 0.787] | 0.767 [0.740, 0.795] | 0.768 [0.745, 0.789] |
| Text + confidence | internal | 0.685 | 0.697 [0.668, 0.725] | 0.618 [0.574, 0.659] | 0.715 [0.686, 0.741] | 0.710 [0.688, 0.733] |
| Text + confidence + global | internal | 0.742 | 0.745 [0.715, 0.774] | 0.742 [0.692, 0.790] | 0.785 [0.759, 0.809] | 0.781 [0.760, 0.803] |
| Text + confidence + region | internal | 0.737 | 0.727 [0.700, 0.757] | 0.735 [0.694, 0.775] | 0.755 [0.728, 0.783] | 0.757 [0.735, 0.778] |
| Full: text + confidence + global + region | internal | 0.742 | 0.730 [0.702, 0.759] | 0.739 [0.688, 0.784] | 0.772 [0.745, 0.800] | 0.772 [0.749, 0.793] |
| Full with SHUFFLED regions (control) | internal, control | 0.717 | 0.718 [0.688, 0.747] | 0.717 [0.668, 0.763] | 0.767 [0.741, 0.794] | 0.763 [0.739, 0.784] |
| Full, gradient boosting (per-layer summaries) | internal | 0.724 | 0.726 [0.700, 0.750] | 0.726 [0.688, 0.761] | 0.749 [0.723, 0.775] | 0.743 [0.720, 0.766] |
| Uncertainty at the finding word (U) | internal | 0.604 | 0.599 [0.570, 0.630] | 0.523 [0.479, 0.569] | 0.615 [0.585, 0.646] | 0.607 [0.583, 0.631] |
| Counterfactual: named region masked (Xn) | internal | 0.501 | 0.533 [0.500, 0.565] | 0.470 [0.422, 0.511] | 0.529 [0.497, 0.562] | 0.519 [0.494, 0.543] |
| Counterfactual: WRONG region masked (control) | internal, control | 0.505 | 0.533 [0.503, 0.565] | 0.517 [0.465, 0.566] | 0.498 [0.466, 0.530] | 0.499 [0.471, 0.527] |
| Counterfactual: blank image (Xb) | internal | 0.567 | 0.619 [0.590, 0.650] | 0.642 [0.598, 0.682] | 0.621 [0.592, 0.653] | 0.604 [0.581, 0.629] |
| Counterfactual, all passes (X) | internal | 0.564 | 0.616 [0.587, 0.647] | 0.632 [0.590, 0.675] | 0.619 [0.590, 0.649] | 0.606 [0.583, 0.630] |
| Finding-word attention: global + region (Gm + Rmn) | internal | 0.742 | 0.726 [0.698, 0.754] | 0.756 [0.718, 0.795] | 0.754 [0.726, 0.780] | 0.758 [0.736, 0.779] |
| Finding-word attention, SHUFFLED regions (control) | internal, control | 0.724 | 0.727 [0.699, 0.755] | 0.751 [0.707, 0.790] | 0.759 [0.732, 0.786] | 0.758 [0.736, 0.778] |
| Text + confidence + uncertainty + counterfactual | internal | 0.713 | 0.714 [0.684, 0.744] | 0.652 [0.610, 0.697] | 0.734 [0.706, 0.762] | 0.734 [0.711, 0.758] |
| Internal v5: everything model-internal | internal | 0.746 | 0.726 [0.699, 0.754] | 0.743 [0.703, 0.783] | 0.771 [0.744, 0.797] | 0.773 [0.752, 0.794] |
| Internal v5 with no named-region information (control) | internal, control | 0.726 | 0.731 [0.704, 0.758] | 0.739 [0.697, 0.781] | 0.773 [0.747, 0.800] | 0.769 [0.747, 0.790] |
| Internal v5, gradient boosting (summaries) | internal | 0.732 | 0.725 [0.700, 0.752] | 0.745 [0.709, 0.781] | 0.747 [0.719, 0.773] | 0.746 [0.722, 0.767] |
| External verifier only (CXR classifier) | external | 0.604 | 0.611 [0.578, 0.646] | 0.616 [0.569, 0.662] | 0.624 [0.589, 0.659] | 0.623 [0.595, 0.648] |
| Text + verifier | external | 0.708 | 0.718 [0.689, 0.747] | 0.675 [0.628, 0.719] | 0.732 [0.704, 0.761] | 0.728 [0.704, 0.751] |
| Text + confidence + verifier | external | 0.707 | 0.717 [0.688, 0.745] | 0.667 [0.625, 0.712] | 0.739 [0.711, 0.766] | 0.733 [0.710, 0.756] |
| Internal v5 + verifier | external | 0.746 | 0.726 [0.700, 0.754] | 0.743 [0.703, 0.784] | 0.772 [0.745, 0.798] | 0.773 [0.752, 0.794] |
| Internal v5 + verifier, gradient boosting | external | 0.736 | 0.740 [0.712, 0.765] | 0.760 [0.724, 0.796] | 0.755 [0.729, 0.781] | 0.751 [0.727, 0.773] |
| Ensemble (internal only): top-3 by calibration CV **[selected internal]** **[selected with verifier]** | internal | 0.751 | 0.740 [0.711, 0.767] | 0.748 [0.700, 0.793] | 0.782 [0.756, 0.809] | 0.781 [0.760, 0.802] |
| Ensemble (with verifier): top-3 by calibration CV | external | 0.747 | 0.730 [0.703, 0.759] | 0.745 [0.702, 0.787] | 0.774 [0.748, 0.800] | 0.775 [0.753, 0.796] |

## Paired comparisons (AUROC difference A - B, image bootstrap, one-sided p)

| question | A vs B | test | test, clean | cross-fit test |
|---|---|---|---|---|
| attention adds over text + confidence | Full: text + confidence + global + region vs Text + confidence | +0.033 [-0.001, +0.065], p=0.033 | +0.121 [+0.065, +0.176], p=0.001 | +0.057 [+0.031, +0.085], p=0.001 |
| full detector vs text-only prior | Full: text + confidence + global + region vs Text only | +0.034 [-0.000, +0.066], p=0.028 | +0.116 [+0.057, +0.173], p=0.001 | +0.060 [+0.035, +0.089], p=0.001 |
| named region matters inside full detector | Full: text + confidence + global + region vs Full with SHUFFLED regions (control) | +0.012 [-0.007, +0.029], p=0.104 | +0.021 [-0.003, +0.047], p=0.045 | +0.004 [-0.010, +0.017], p=0.268 |
| named region matters (attention only) | Region alignment (R, VERA-probe) vs Region alignment, SHUFFLED regions (control) | +0.075 [+0.043, +0.108], p=0.001 | +0.089 [+0.049, +0.134], p=0.001 | +0.065 [+0.036, +0.093], p=0.001 |
| region adds over global attention | Global + region (attention only) vs Global attention, no region (G) | -0.013 [-0.028, +0.002], p=0.959 | -0.004 [-0.025, +0.015], p=0.651 | -0.011 [-0.024, +0.001], p=0.960 |
| region adds over text + confidence | Text + confidence + region vs Text + confidence | +0.030 [-0.001, +0.061], p=0.031 | +0.117 [+0.074, +0.156], p=0.001 | +0.041 [+0.015, +0.069], p=0.004 |
| claim-level beats image-level attention | Global attention, no region (G) vs Image-level attention only (control) | +0.057 [+0.030, +0.087], p=0.001 | +0.023 [-0.036, +0.078], p=0.192 | +0.085 [+0.061, +0.109], p=0.001 |
| masking the NAMED region matters more than masking the wrong region (causal grounding) | Counterfactual: named region masked (Xn) vs Counterfactual: WRONG region masked (control) | -0.001 [-0.042, +0.040], p=0.508 | -0.047 [-0.106, +0.011], p=0.943 | +0.031 [-0.003, +0.067], p=0.043 |
| finding-word attention beats sentence-averaged attention | Finding-word attention: global + region (Gm + Rmn) vs Global + region (attention only) | -0.002 [-0.022, +0.019], p=0.586 | +0.016 [-0.026, +0.071], p=0.305 | -0.013 [-0.028, +0.003], p=0.945 |
| v5 signals add over v4 | Internal v5: everything model-internal vs Full: text + confidence + global + region | -0.004 [-0.015, +0.006], p=0.784 | +0.004 [-0.014, +0.027], p=0.368 | -0.000 [-0.011, +0.009], p=0.560 |
| everything internal vs text + confidence | Internal v5: everything model-internal vs Text + confidence | +0.029 [-0.004, +0.059], p=0.040 | +0.125 [+0.080, +0.172], p=0.001 | +0.057 [+0.032, +0.084], p=0.001 |
| named region matters inside v5 | Internal v5: everything model-internal vs Internal v5 with no named-region information (control) | -0.006 [-0.024, +0.013], p=0.720 | +0.004 [-0.020, +0.028], p=0.376 | -0.002 [-0.017, +0.013], p=0.617 |
| VLM-internal signals add over an external classifier | Internal v5 + verifier vs Text + confidence + verifier | +0.009 [-0.019, +0.039], p=0.285 | +0.076 [+0.033, +0.118], p=0.002 | +0.033 [+0.009, +0.060], p=0.005 |
| external classifier adds over internal signals | Internal v5 + verifier vs Internal v5: everything model-internal | +0.001 [+0.000, +0.001], p=0.004 | +0.000 [-0.000, +0.001], p=0.137 | +0.001 [-0.000, +0.001], p=0.052 |
| selected internal detector vs text-only prior | Ensemble (internal only): top-3 by calibration CV vs Text only | +0.043 [+0.012, +0.074], p=0.002 | +0.126 [+0.070, +0.180], p=0.001 | +0.071 [+0.046, +0.099], p=0.001 |
| selected internal detector vs text + confidence | Ensemble (internal only): top-3 by calibration CV vs Text + confidence | +0.043 [+0.012, +0.074], p=0.002 | +0.130 [+0.079, +0.182], p=0.001 | +0.068 [+0.044, +0.095], p=0.001 |
| selected overall detector vs verifier alone | Ensemble (internal only): top-3 by calibration CV vs External verifier only (CXR classifier) | +0.129 [+0.088, +0.170], p=0.001 | +0.133 [+0.078, +0.191], p=0.001 | +0.158 [+0.122, +0.198], p=0.001 |
| selected internal vs training-free | Ensemble (internal only): top-3 by calibration CV vs training-free VERA (v3 primary) | +0.143 [+0.106, +0.181], p=0.001 | +0.134 [+0.060, +0.208], p=0.001 | +0.185 [+0.151, +0.222], p=0.001 |

## Selected internal model per tier (test)
- critical: 0.751 [0.692, 0.811] (n=303); clean labels 0.713 [0.622, 0.799] (n=201)
- moderate: 0.724 [0.684, 0.760] (n=815); clean labels 0.788 [0.739, 0.834] (n=439)
- mild: 0.773 [0.713, 0.829] (n=274); clean labels 0.758 [0.654, 0.851] (n=146)

## Selected model with verifier, per tier (test)
- critical: 0.751 [0.692, 0.811] (n=303)
- moderate: 0.724 [0.684, 0.760] (n=815)
- mild: 0.773 [0.713, 0.829] (n=274)

## Operating points, selected internal (thresholds from calibration)
- flag 10%: test flag rate 10.3%, precision 0.840, recall 0.146, F1 0.249
- flag 20%: test flag rate 20.8%, precision 0.827, recall 0.289, F1 0.428
- flag 30%: test flag rate 26.9%, precision 0.819, recall 0.371, F1 0.510
- flag 50%: test flag rate 48.3%, precision 0.783, recall 0.635, F1 0.701

## Operating points, selected with verifier (thresholds from calibration)
- flag 10%: test flag rate 10.3%, precision 0.840, recall 0.146, F1 0.249
- flag 20%: test flag rate 20.8%, precision 0.827, recall 0.289, F1 0.428
- flag 30%: test flag rate 26.9%, precision 0.819, recall 0.371, F1 0.510
- flag 50%: test flag rate 48.3%, precision 0.783, recall 0.635, F1 0.701

## Counterfactual masking: mean log-prob drop of the finding word (real minus masked)
- cf_d_blank_m: supported +0.185, hallucinated +0.239, AUROC of (low drop = risk) 0.489
- cf_d_named_m: supported +0.165, hallucinated +0.235, AUROC of (low drop = risk) 0.482
- cf_d_ctrl_m: supported +0.062, hallucinated +0.100, AUROC of (low drop = risk) 0.485
- cf_named_minus_ctrl_m: supported +0.103, hallucinated +0.136, AUROC of (low drop = risk) 0.474

Share of |coefficients| per family: {'Full: text + confidence + global + region': {'T': 0.028744962066411972, 'C': 0.0026911168824881315, 'G': 0.3878406286239624, 'R': 0.5807232856750488}, 'Internal v5: everything model-internal': {'T': 0.01382763683795929, 'C': 0.0008936601807363331, 'U': 0.0008783851517364383, 'X': 0.00149331905413419, 'G': 0.18892109394073486, 'R': 0.28654831647872925, 'Gm': 0.2128838151693344, 'Rmn': 0.2945537567138672}, 'Internal v5 + verifier': {'T': 0.013813577592372894, 'C': 0.0008941954001784325, 'U': 0.0008760523051023483, 'X': 0.0014956925297155976, 'G': 0.18846724927425385, 'R': 0.2859615683555603, 'Gm': 0.2123943418264389, 'Rmn': 0.2941499650478363, 'V': 0.0019472900312393904}}