# Model selection log

All decisions below were made on the validation split. The test split has not been used for any decision.

Rule, declared before the seed runs: rank configs by final-epoch QWK, break ties with macro-F1, and report rare-grade recall alongside. Differences smaller than the seed-to-seed spread are treated as ties.

Final epoch, mean (sd) over seeds 42, 43, 44, validation split:

| Config | QWK | Accuracy | Macro-F1 |
|---|---|---|---|
| 512 px, no class weights | 0.918 (0.002) | 0.881 (0.001) | 0.759 (0.012) |
| 512 px, weights^0.5 | 0.919 (0.009) | 0.876 (0.011) | 0.757 (0.012) |
| 512 px, full class weights | 0.908 (0.002) | 0.865 (0.000) | 0.735 (0.004) |
| 256 px, no class weights | 0.902 (0.004) | 0.849 (0.007) | 0.690 (0.019) |

Findings:
- Resolution: 512 px beats 256 px (QWK +0.016, accuracy +3.2 points, macro-F1 +6.9 points); per-seed ranges do not overlap.
- Class weighting at 512 px: full weighting is lowest; no weights and weights^0.5 are tied.
- Enhancement (Ben Graham) and 384 px were tried with one seed each and not pursued; the ordinal regression head scored lower (QWK 0.892) with one seed.

Decision (2026-11-13): primary model is EfficientNet-B0, 512 px, no class weights. weights^0.5 is reported as the tied comparison.

## Final model and test protocol (declared 2026-11-14, before any test-split evaluation)

Validation, final epoch, 3-seed ensembles (average of softmax probabilities):

| Model | QWK | Accuracy | Macro-F1 | Referable AUC | False alarms at 95% sens |
|---|---|---|---|---|---|
| 512 px, no class weights | 0.928 | 0.897 | 0.789 | 0.994 | 12 |
| 512 px, weights^0.5 | 0.934 | 0.895 | 0.794 | 0.992 | 20 |

Both ensembles beat their best single member on QWK, accuracy and macro-F1. They tie on those metrics; the no-class-weight ensemble has the better referable-DR trade-off (higher AUC, fewer false alarms at equal sensitivity).

Final model: average of the softmax probabilities of the three 512 px, no-class-weight EfficientNet-B0 runs (seeds 42, 43, 44).

Referable-DR operating point: target sensitivity >= 0.95, with the cut-off on P(grade >= 2) chosen on the validation split and then applied unchanged to test.

The test split is evaluated once, for these models only:
- the final model (3-seed ensemble)
- the weights^0.5 3-seed ensemble (tied comparison)
- the 256 px baseline (EfficientNet-B0, full class weights, seed 42)
- the classical baseline (gradient boosting on handcrafted features)

No model, setting or cut-off will be changed after seeing test results; results are reported as obtained.
