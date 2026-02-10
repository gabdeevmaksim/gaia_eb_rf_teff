# Test sample metrics: MAE, RMSE, R²

Summary of test-set performance for all four Teff prediction models.

| Model | MAE (K) | RMSE (K) | R² | N (test) | % chosen as best |
|-------|--------|----------|-----|----------|-------------------|
| Clustering (corrected, Optuna) | 585.9 | 1177.3 | 0.6498 | 255,234 | 117,247 (13.8%) |
| Log (corrected, Optuna) | 585.0 | 1173.6 | 0.6520 | 255,234 | 190,436 (22.5%) |
| Flag1 (corrected, Optuna) | 191.0 | 283.4 | 0.9066 | 142,334 | 497,391 (58.7%) |
| Chain (logg → Teff) | 588.4 | 1066.1 | 0.6081 | 255,234 | 42,412 (5.0%) |

**% chosen as best:** Among objects with at least one prediction in the unified catalog, the number and fraction for which this model had the lowest uncertainty and was therefore selected as `teff_best` (format: count, percentage).
