# VERA results: training-free score, baselines, v3 probes

- Primary claims: calib 960, test 1392 (837 test images); hallucination prevalence test = 0.5948275862068966
- Visual grid 32x32, 32 layers x 32 heads; segmentation masks used for 0.997874149659864 of primary claims
- Flag-all F1 on test: 0.745945945945946

## Primary (pre-specified)
- seg:lift, layer 9 (selected on calibration): AUROC 0.597 [0.567, 0.628], calib AUROC 0.610
- Head ensemble: AUROC 0.622 [0.592, 0.652]

## Calibration-trained probes (test AUROC, 95% CI)
- Probe: region alignment, all heads (VERA-probe): 0.723 [0.695, 0.752] (calib CV 0.705, C=0.001)
- Probe: region alignment with SHUFFLED regions (control): 0.661 [0.632, 0.689] (calib CV 0.669, C=0.003)
- Probe: image-attention share + confidence, no region (control): 0.751 [0.723, 0.778] (calib CV 0.732, C=0.001)
- Probe: region alignment + share + confidence: 0.740 [0.714, 0.768] (calib CV 0.725, C=0.0003)

## Baselines (test AUROC, 95% CI)
- Confidence: mean token log-prob: 0.553 [0.521, 0.588]
- Confidence: min token log-prob: 0.542 [0.509, 0.576]
- Visual reliance (attention share to image, all layers/heads): 0.587 [0.555, 0.617]
- Attention entropy (layer 9): 0.451 [0.418, 0.485]
- v1 recipe (rect, raw, last-4 layers, whole-report map): 0.545 [0.514, 0.578]
- Random: 0.471 [0.439, 0.503]

## Ablations (test AUROC, 95% CI)
- seg:raw @ calib layer 9: 0.628 [0.598, 0.659]
- seg:lift @ calib layer 9: 0.597 [0.567, 0.628]
- seg:excess @ calib layer 9: 0.638 [0.609, 0.668]
- rect:raw @ calib layer 9: 0.624 [0.595, 0.656]
- rect:lift @ calib layer 9: 0.597 [0.567, 0.625]
- rect:excess @ calib layer 9: 0.620 [0.590, 0.650]
- seg:lift, last-4-layer average (v1 layer choice): 0.550 [0.518, 0.581]
- seg:lift, whole-report map instead of sentence (layer 9): 0.617 [0.587, 0.646]

## Per tier
- critical: 0.680 [0.618, 0.736] (n=303)
- moderate: 0.571 [0.529, 0.612] (n=815)
- mild: 0.594 [0.520, 0.655] (n=274)

## Operating points (primary, thresholds from calibration)
- target flag rate 10%: test flag rate 7.0%, recall 0.098, precision 0.835, F1 0.175
- target flag rate 20%: test flag rate 17.0%, recall 0.222, precision 0.780, F1 0.346
- target flag rate 30%: test flag rate 27.5%, recall 0.331, precision 0.715, F1 0.453
- target flag rate 50%: test flag rate 45.8%, recall 0.513, precision 0.667, F1 0.580

## Secondary claim types (same score, test)
- negated_claims: 0.491 [0.450, 0.536] (n=1776)
- normal_claims: 0.605 [0.574, 0.635] (n=1518)
- hedged_claims: 0.540 [0.446, 0.638] (n=295)

Primary label reasons: {np.str_('absent_in_ref'): 1267, np.str_('mislocalized'): 30, np.str_('negated_in_ref'): 145, np.str_('supported'): 910}