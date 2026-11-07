# VERA results: training-free score, baselines, v3 probes

- Primary claims: calib 806, test 1187 (777 test images); hallucination prevalence test = 0.5248525695029486
- Visual grid 32x32, 32 layers x 32 heads; segmentation masks used for 0.9979929754139488 of primary claims
- Flag-all F1 on test: 0.6883977900552486

## Primary (pre-specified)
- seg:lift, layer 9 (selected on calibration): AUROC 0.577 [0.539, 0.610], calib AUROC 0.604
- Head ensemble: AUROC 0.599 [0.565, 0.633]

## Calibration-trained probes (test AUROC, 95% CI)
- Probe: region alignment, all heads (VERA-probe): 0.723 [0.691, 0.753] (calib CV 0.724, C=0.001)
- Probe: region alignment with SHUFFLED regions (control): 0.661 [0.630, 0.689] (calib CV 0.688, C=0.001)
- Probe: image-attention share + confidence, no region (control): 0.756 [0.727, 0.784] (calib CV 0.761, C=0.001)
- Probe: region alignment + share + confidence: 0.749 [0.718, 0.777] (calib CV 0.751, C=0.001)

## Baselines (test AUROC, 95% CI)
- Confidence: mean token log-prob: 0.520 [0.486, 0.556]
- Confidence: min token log-prob: 0.509 [0.476, 0.543]
- Visual reliance (attention share to image, all layers/heads): 0.562 [0.525, 0.597]
- Attention entropy (layer 9): 0.458 [0.424, 0.492]
- v1 recipe (rect, raw, last-4 layers, whole-report map): 0.560 [0.523, 0.591]
- Random: 0.475 [0.442, 0.510]

## Ablations (test AUROC, 95% CI)
- seg:raw @ calib layer 9: 0.636 [0.599, 0.668]
- seg:lift @ calib layer 9: 0.577 [0.539, 0.610]
- seg:excess @ calib layer 9: 0.631 [0.596, 0.663]
- rect:raw @ calib layer 9: 0.621 [0.587, 0.651]
- rect:lift @ calib layer 8: 0.566 [0.532, 0.598]
- rect:excess @ calib layer 9: 0.597 [0.560, 0.627]
- seg:lift, last-4-layer average (v1 layer choice): 0.546 [0.513, 0.578]
- seg:lift, whole-report map instead of sentence (layer 9): 0.613 [0.579, 0.647]

## Per tier
- critical: 0.576 [0.501, 0.654] (n=208)
- moderate: 0.561 [0.515, 0.610] (n=711)
- mild: 0.609 [0.527, 0.677] (n=268)

## Operating points (primary, thresholds from calibration)
- target flag rate 10%: test flag rate 8.3%, recall 0.112, precision 0.714, F1 0.194
- target flag rate 20%: test flag rate 17.9%, recall 0.213, precision 0.627, F1 0.319
- target flag rate 30%: test flag rate 27.7%, recall 0.318, precision 0.602, F1 0.416
- target flag rate 50%: test flag rate 45.8%, recall 0.496, precision 0.568, F1 0.530

## Secondary claim types (same score, test)
- negated_claims: 0.420 [0.282, 0.565] (n=1860)
- normal_claims: 0.451 [0.403, 0.491] (n=1518)
- hedged_claims: 0.575 [0.478, 0.686] (n=226)

Primary label reasons: {np.str_('absent_in_ref'): 935, np.str_('mislocalized'): 30, np.str_('negated_in_ref'): 121, np.str_('supported'): 863, np.str_('supported_by_child'): 44}