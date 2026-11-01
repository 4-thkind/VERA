# VERA v3 results

- Primary claims: calib 871, test 1304 (765 test images); hallucination prevalence test = 0.5858895705521472
- Visual grid 32x32, 32 layers x 32 heads; segmentation masks used for 0.9977011494252873 of primary claims
- Flag-all F1 on test: 0.7388781431334622

## Primary (pre-specified)
- seg:lift, layer 9 (selected on calibration): AUROC 0.598 [0.567, 0.630], calib AUROC 0.615
- Head ensemble: AUROC 0.630 [0.600, 0.661]

## Calibration-trained probes (test AUROC, 95% CI)
- Probe: region alignment, all heads (VERA-probe): 0.708 [0.680, 0.736] (calib CV 0.712, C=0.001)
- Probe: region alignment with SHUFFLED regions (control): 0.660 [0.628, 0.693] (calib CV 0.674, C=0.001)
- Probe: image-attention share + confidence, no region (control): 0.756 [0.728, 0.788] (calib CV 0.750, C=0.001)
- Probe: region alignment + share + confidence: 0.745 [0.717, 0.775] (calib CV 0.733, C=0.001)

## Baselines (test AUROC, 95% CI)
- Confidence: mean token log-prob: 0.529 [0.493, 0.565]
- Confidence: min token log-prob: 0.522 [0.487, 0.557]
- Visual reliance (attention share to image, all layers/heads): 0.602 [0.567, 0.636]
- Attention entropy (layer 9): 0.478 [0.443, 0.517]
- v1 recipe (rect, raw, last-4 layers, whole-report map): 0.527 [0.492, 0.560]
- Random: 0.493 [0.462, 0.528]

## Ablations (test AUROC, 95% CI)
- seg:raw @ calib layer 9: 0.621 [0.589, 0.655]
- seg:lift @ calib layer 9: 0.598 [0.567, 0.630]
- seg:excess @ calib layer 9: 0.631 [0.600, 0.662]
- rect:raw @ calib layer 9: 0.611 [0.577, 0.644]
- rect:lift @ calib layer 9: 0.591 [0.558, 0.625]
- rect:excess @ calib layer 9: 0.609 [0.577, 0.641]
- seg:lift, last-4-layer average (v1 layer choice): 0.552 [0.520, 0.584]
- seg:lift, whole-report map instead of sentence (layer 9): 0.611 [0.580, 0.642]

## Per tier
- critical: 0.695 [0.636, 0.758] (n=278)
- moderate: 0.587 [0.545, 0.629] (n=748)
- mild: 0.549 [0.475, 0.621] (n=278)

## Operating points (primary, thresholds from calibration)
- target flag rate 10%: test flag rate 9.6%, recall 0.126, precision 0.768, F1 0.216
- target flag rate 20%: test flag rate 19.8%, recall 0.242, precision 0.717, F1 0.362
- target flag rate 30%: test flag rate 29.9%, recall 0.348, precision 0.682, F1 0.461
- target flag rate 50%: test flag rate 49.8%, recall 0.556, precision 0.655, F1 0.602

## Secondary claim types (same score, test)
- negated_claims: 0.461 [0.415, 0.506] (n=1503)
- normal_claims: 0.600 [0.568, 0.634] (n=1266)
- hedged_claims: 0.565 [0.478, 0.655] (n=252)

Primary label reasons: {np.str_('absent_in_ref'): 1142, np.str_('mislocalized'): 29, np.str_('negated_in_ref'): 129, np.str_('supported'): 875}