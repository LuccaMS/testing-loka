### 5. ALT Regression
**Notebook:** `notebooks/regression.ipynb`

A XGBoost Regressor was trained to predict `alanine_aminotransferase` (ALT).

#### Statistical Analysis
Pearson correlation and ANOVA were used to assess feature relationships with ALT:

| Feature | Test | Statistic | p-value | Significant |
|---------|------|-----------|---------|-------------|
| bmi | Pearson r | 0.9998 | < 0.0001 | ✅ |
| readmitted | ANOVA F | 6.683 | 0.0097 | ✅ |
| All other features | — | — | > 0.05 | ❌ |

> `readmitted` is statistically significant but practically irrelevant —
> the mean ALT difference between readmitted and non-readmitted patients
> is only **0.29 units (0.85% of the total ALT range)**.

#### Model Performance (5-Fold Cross-Validation)
| Metric | Score |
|--------|-------|
| R² | 0.9990 ± 0.0001 |
| RMSE | 0.1513 ± 0.0112 |
| MAE | 0.0956 ± 0.0021 |

#### Feature Importance
| Feature | Importance |
|---------|------------|
| bmi | 99.47% |
| All other features | 0.53% |

> ⚠️ The model achieves near-perfect performance (R² = 0.9990), but this is
> a synthetic data artifact — ALT ≈ BMI + small random noise (r = 0.9998).
> In real clinical data, ALT would depend on a broader set of features such
> as liver function indicators, alcohol consumption, and medication history.

#### Conclusion
The regression model works and demonstrates technical capability, but the
underlying data relationship is trivial. **In production, this model would
require real clinical data to be meaningful.**