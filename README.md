# Diabetes Risk Estimation in Mexican Patients

**A machine learning model to decide which patients should get a confirmatory diabetes test first, without using glucose or HbA1c.**

## The problem

In Mexico, the tests that confirm diabetes (fasting glucose or HbA1c) are a limited resource in many health programs. This project builds a model that **ranks adult patients by their probability of being high risk**, using data that is often already available: age, sex, weight, height and a biochemical profile (lipids, insulin, uric acid, creatinine, albumin).

The model **does not diagnose**: it helps prioritize who to see first, always with review by healthcare staff.

## Results

Final model: **SVM with an RBF kernel** (C = 3, gamma = 0.03), with 15 features and a decision threshold of **0.26**, chosen to detect at least 80% of high-risk cases.

Evaluation on a test set of **365 patients from households the model never saw**:

| Metric (high risk vs. rest) | Value |
| --- | --- |
| **Recall** | **0.825** (detects 99 of 120 high-risk cases) |
| ROC-AUC | 0.765 |
| Precision | 0.481 |
| F1-score | 0.607 |
| PR-AUC | 0.588 |
| Cross-validation ROC-AUC | 0.764 +/- 0.027 (15 folds) |

The baseline model (`DummyClassifier`) gets a ROC-AUC of 0.505. The test metric falls within the cross-validation range, with no signs of overfitting.

![ROC and precision-recall curves on the test set](docs/img/roc_pr_prueba.png)

## How it was built

### 1. Data and data leakage
- **2,549 adults from 2,055 households**, with 24 variables from an ENSANUT-style national health survey ([Diabetes Mexico Data Set, Kaggle](https://doi.org/10.34740/KAGGLE/DSV/14619160)).
- Target variable `riesgo_diabetes_cat`: low (66.6%), moderate (0.5%) and high (33.0%).
- **Glucose and HbA1c were excluded**, because they are the variables used to build the target: using them would have produced a model that looks almost perfect on paper and is useless in practice.
- **Sentinel values** (for example, weight = 222.22) that produced fake BMI values were detected and cleaned.

![Target distribution and glucose by risk level](docs/img/objetivo_y_fuga.png)

### 2. Leak-free validation
- Split with `StratifiedGroupKFold` **grouped by household**: members of the same family share habits, so they are never split between training and test.
- Repeated cross-validation (5 folds x 3 seeds = 15 evaluations).
- The test set was used **only once**, at the end, to evaluate the finalists.

### 3. Feature engineering
Four new features with clinical meaning and no leakage: triglyceride/HDL ratio (an indicator of insulin resistance), non-HDL cholesterol, age group and BMI category.

### 4. Model comparison
8 models were compared within the same scikit-learn pipeline, with 19 logged experiments:

| Model | ROC-AUC (high risk, CV) |
| --- | --- |
| **RBF SVM** | **0.764** |
| Random Forest | 0.756 |
| Linear SVM | 0.752 |
| Gradient Boosting | 0.746 |
| Logistic regression | 0.723 |
| Neural network (MLP) | 0.720 |
| Decision tree | 0.674 |
| DummyClassifier | 0.505 |

Hyperparameters were tuned with `GridSearchCV` (SVM) and `RandomizedSearchCV` (Random Forest). The final model was chosen with a **weighted decision matrix** (recall, precision, ROC-AUC, stability, interpretability and operating cost), not just by the highest metric.

![Model comparison](docs/img/comparacion_modelos.png)

### 5. Decision threshold
In healthcare, missing a high-risk patient costs more than an extra test. That is why the threshold was optimized with an explicit criterion: **minimum recall of 0.80**, using out-of-fold predictions (without touching the test set).

### 6. Interpretability
Permutation importance and SHAP. The most influential features are **age, insulin and triglycerides**, in line with known metabolic risk factors.

![SHAP values of the final model](docs/img/shap_global.png)

### 7. Bias and responsible use
- Even performance by sex: recall of 0.83 for men and 0.82 for women.
- **Important limitation:** in adults aged 20 to 39, recall drops to 0.47, because high risk is less frequent and its markers are less clear. This is documented in the model card.
- The "moderate" class has only 12 cases and the model does not detect it.

![Metrics by group](docs/img/sesgos.png)

### 8. Ready to operate
- Full pipeline serialized with `joblib`, with a SHA-256 hash to verify its integrity.
- Standalone inference script with **input validation** (missing fields, wrong types, unknown categories, out-of-range values).
- Monitoring plan with **PSI** (Population Stability Index) to detect data drift, plus performance and fairness alerts.
- Model card with intended uses, non-recommended uses, risks and update criteria.

## How to run it

1. Download the dataset from [Kaggle](https://doi.org/10.34740/KAGGLE/DSV/14619160) and save `Diabetes_Mexico.csv` next to the notebook (or in `data/`).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `Proyecto_Final_Riesgo_Diabetes.ipynb` from start to finish (it also works in Google Colab).

When it runs, the notebook automatically generates:

```
scripts/     Custom transformers and inference script
artifacts/   Serialized pipeline, threshold and model metadata
results/     Experiment log, metrics and figures
docs/        Model card
```

## Tech stack

Python, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Seaborn and Joblib.

## Author

**Alan Adelfo Del Angel Figueroa**, Software Development Engineering student at Universidad Tecmilenio.
Final project for the Machine Learning Algorithms course (2026).

- Portfolio and contact: [github.com/AlanDelAngel](https://github.com/AlanDelAngel)

## References

- Félix Jiménez, A. F., & Sánchez Lee, V. S. (2026). *Diabetes Mexico Data Set* [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/14619160
- American Diabetes Association. (2023). Classification and diagnosis of diabetes: Standards of Care in Diabetes 2023. *Diabetes Care, 46*(Supplement 1), S19-S40. https://doi.org/10.2337/dc23-S002
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

---

### Resumen en español

Modelo de machine learning que prioriza a qué pacientes adultos mexicanos conviene hacerles primero una prueba confirmatoria de diabetes, **sin usar glucosa ni HbA1c** (las variables que definen el objetivo, excluidas para evitar fuga de información). Se construyó con 2,549 pacientes de una encuesta nacional de salud, con validación cruzada agrupada por hogar, 8 modelos comparados y un umbral de decisión ajustado para el recall clínico. La SVM RBF final alcanza **un recall de 0.825 y un ROC-AUC de 0.765** en un conjunto de prueba de hogares nunca vistos, e incluye interpretabilidad con SHAP, análisis de sesgos por subgrupo, un pipeline serializado con validación de entradas, un plan de monitoreo con PSI y una ficha del modelo.
