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
