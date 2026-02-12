## Methodology

### 1. Feature Engineering
The raw dataset was preprocessed in `notebooks/feature_engineering.ipynb`, where categorical 
variables were encoded with proper ordinal relationships and missing values were preserved 
for XGBoost's native NaN handling.

---

### 2. COPD Classification
**Notebook:** `notebooks/classifier.ipynb`

A XGBoost Classifier was trained to predict `chronic_obstructive_pulmonary_disease` (classes A, B, C, D).
To ensure robustness, **3 algorithms** and **127+ hyperparameter configurations** were tested.

#### Results (5-Fold Cross-Validation)
| Metric | Score |
|--------|-------|
| Accuracy | 0.2598 ± 0.0045 |
| F1 (Macro) | 0.2597 ± 0.0046 |
| Precision | 0.2598 ± 0.0046 |
| Recall | 0.2597 ± 0.0045 |

---

### 3. Statistical Investigation
**Notebook:** `notebooks/analysis_classifier.ipynb`

Given the poor model performance, ANOVA tests were conducted to check whether 
any feature had a statistically significant relationship with the COPD classes.

| Feature | F-statistic | p-value | Significant |
|---------|-------------|---------|-------------|
| age | 0.589 | 0.6221 | ❌ |
| bmi | 0.209 | 0.8901 | ❌ |
| medication_count | 0.563 | 0.6395 | ❌ |
| days_hospitalized | 0.492 | 0.6881 | ❌ |
| last_lab_glucose | 1.663 | 0.1726 | ❌ |
| albumin_globulin_ratio | 0.999 | 0.3920 | ❌ |
| sex | 0.471 | 0.7027 | ❌ |
| smoker | 0.403 | 0.7507 | ❌ |
| readmitted | 0.420 | 0.7387 | ❌ |
| urban | 0.154 | 0.9269 | ❌ |
| exercise_frequency | 3.624 | 0.0125 | ✅ |
| diet_quality | 1.189 | 0.3122 | ❌ |
| income_bracket | 0.851 | 0.4659 | ❌ |
| education_level | 1.689 | 0.1670 | ❌ |
| diagnosis_code | 1.528 | 0.2051 | ❌ |

**14/15 features showed no significant relationship with COPD (p > 0.05).**

`exercise_frequency` was the only exception, so a model using it alone was tested:

| Model | Accuracy |
|-------|----------|
| All features | 0.2545 ± 0.0077 |
| exercise_frequency only | 0.2463 ± 0.0060 |
| Random baseline | 0.2500 |

The single-feature model performed **below random guessing**, confirming that the 
apparent significance of `exercise_frequency` was not meaningful in practice.

---

### 4. Conclusion

> The COPD labels show no relationship with any feature in the dataset.
> This is consistent with random label assignment during synthetic data generation.
> **A COPD classification model cannot be built from this dataset.**

