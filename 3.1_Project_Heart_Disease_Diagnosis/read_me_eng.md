# Heart Disease Classification: Manual Feature Engineering vs. OpenFE

[Bản tiếng Việt](read_me_vn.md)

This project reproduces and extends a Cleveland heart-disease classification pipeline. It compares preprocessed baseline features, manually engineered features, and automated OpenFE features using the same data splits and evaluation models.

The highest recorded test accuracy is **90.32% (28/31 correct predictions)** from stacking with manual FE. KNN with tuned OpenFE features achieves **87.10% (27/31)**. These are results on one fixed split, rather than evidence of generalization to new clinical populations.

## 1. Dataset and task

- **303 samples**, **13 input features**, and one target column.
- Source: the Cleveland subset of [Heart Disease — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
- Input file: [data/cleveland.csv](data/cleveland.csv), without a header row.
- There are 4 missing entries in `ca` and 2 in `thal`; the `?` marker is converted to missing values during processing.
- Final task: binary classification, **0 = no disease**, **1 = disease present**.

### Cleveland Heart Disease feature dictionary

| Feature | Meaning and encoding |
| --- | --- |
| `age` | Patient age, measured in years. |
| `sex` | Sex indicator: `1` for male and `0` for female. |
| `cp` | Chest-pain category: `1` typical angina; `2` atypical angina; `3` non-cardiac chest pain; `4` asymptomatic. |
| `trestbps` | Blood pressure measured at rest on admission, in mmHg. |
| `chol` | Serum cholesterol concentration, in mg/dL. |
| `fbs` | Whether fasting glucose exceeds 120 mg/dL: `1` yes, `0` no. |
| `restecg` | Resting ECG category: `0` normal; `1` ST–T abnormalities; `2` left-ventricular hypertrophy under the dataset criteria. |
| `thalach` | Highest heart rate reached, in beats per minute. |
| `exang` | Angina triggered by exercise: `1` present, `0` absent. |
| `oldpeak` | Exercise-related ST depression relative to the resting measurement. |
| `slope` | ST-segment slope at peak exercise: `1` rising; `2` flat; `3` falling. |
| `ca` | Count of major vessels visible through fluoroscopy, ranging from `0` to `3`. |
| `thal` | Thal result code: `3` normal; `6` fixed defect; `7` reversible defect. |
| `num` | Original outcome: `0` indicates no disease; `1–4` indicate disease. The code names this column `target`. |

The `thal` description follows the UCI result codes; this project does not treat it as a diagnosis of thalassemia. The source file retains its original labels, while exported split files apply:

```python
target = (num > 0).astype(int)
```

## 2. Work completed

### Exploratory analysis

- Inspected data types, class counts, and missing values.
- Built target-stacked categorical count charts, numeric histograms, and boxplots.
- Visualized Pearson correlations and scatter plots with regression lines, colored by disease status.

### Splitting and preprocessing

Two stratified `train_test_split` calls use `random_state=42` to obtain an approximate 80:10:10 split:

| Split | Samples | No disease / disease |
| --- | ---: | ---: |
| Train | 242 | 131 / 111 |
| Validation | 30 | 16 / 14 |
| Test | 31 | 17 / 14 |

Imputation, scalers, encoders, and feature selection are fitted on train, then applied to validation/test.

| Representation | Numeric processing | Categorical processing | Final features |
| --- | --- | --- | ---: |
| Baseline `data` | Median imputation, StandardScaler | Mode imputation, MinMaxScaler | 13 |
| Manual `fe_data` | Median imputation, StandardScaler | Mode imputation, One-hot | 13 |
| Tuned `openfe_data` | Median imputation, StandardScaler | Mode imputation, One-hot | 30 |

### Manual feature engineering

The `adding_feature()` function creates `chol/age`, `trestbps/age`, `thalach/age`, and an `age_bin` column using `pd.cut(..., bins=5)`. After preprocessing, constant columns are removed and training mutual information (MI) selects **Top-13** features, matching the original input-feature count.

The pipeline was aligned with the course solution. All six baseline/manual-FE CSVs were verified against the reference data, including values and column ordering.

### Automated feature engineering with OpenFE

- Enumerated **95 first-order numeric candidates**, restricted to row-wise transformations rather than cross-split group/frequency statistics.
- Used `stage1_metric="predictive"`, `n_data_blocks=1`, `min_candidate_features=95`, and LightGBM gain ranking.
- Compared `feature_boosting=False` and `True` using training data.
- Removed constant/duplicate columns; retained the 28 original One-hot/numeric columns and forward-selected generated additions using stratified 5-fold KNN CV accuracy.
- Applied no fixed Top-K: selection stops when another feature no longer improves the CV score.

Both boosting settings reached **85.10% CV accuracy**; `False` was chosen on a tie. Forward selection improved the score from **82.64% to 85.10%** by adding `age/thalach` and `age*oldpeak`, producing **30 final features**. MI remains available for descriptive ranking and visualization, not as an export-count restriction.

### Models and evaluation

| Model | Configuration / selection |
| --- | --- |
| KNN | Search `k=1..20` with shuffled stratified 5-fold CV; maximize mean accuracy. |
| Decision Tree | Search `max_depth=2..5` with the same CV procedure. |
| KMeans | Two clusters with `init="random"`; assign each cluster its majority training class. |
| Gaussian Naive Bayes | Fit `GaussianNB()` to each representation. |
| Stacking | Base models: KNN `k=5`, an unlimited-depth DT, and GaussianNB. Final model: KNN `k=5`, trained on out-of-fold probabilities using **unshuffled** 5-fold CV and `passthrough=False`. |

`evaluate_val.py` and `evaluate_test.py` share accuracy, balanced accuracy, macro precision/recall/F1, and confusion-matrix calculations. Selected standalone KNN values are **11 / 15 / 5** for baseline / manual FE / OpenFE; all three DTs select **depth=3**. Stacking retains its separate reference configuration instead of inheriting standalone tuning results.

## 3. Current results

Accuracy on the **same 31 test samples**:

| Model | Baseline | Manual FE | Tuned OpenFE |
| --- | ---: | ---: | ---: |
| KNN | 83.87% | 83.87% | 87.10% |
| Decision Tree | 80.65% | 80.65% | 80.65% |
| KMeans | 87.10% | 87.10% | 87.10% |
| Gaussian Naive Bayes | 83.87% | 83.87% | 70.97% |
| Stacking | 83.87% | **90.32%** | 80.65% |

| Stacking representation | Validation accuracy | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | 86.67% | 83.87% | 0.8387 |
| Manual FE | 93.33% | 90.32% | 0.9032 |
| Tuned OpenFE | 96.67% | 80.65% | 0.8062 |

An earlier unfiltered OpenFE experiment exported 104 features and achieved 64.52% stacking test accuracy. CV-based selection increased this to 80.65%, but did not exceed manual FE. More features and more complex ensembles did not consistently improve performance across models.

## 4. Main files and execution

| File / directory | Purpose |
| --- | --- |
| [feature_engineering.ipynb](feature_engineering.ipynb) | EDA, baseline/manual FE, MI ranking, and split export. |
| [heart_disease_diagnosis.ipynb](heart_disease_diagnosis.ipynb) | Baseline/manual-FE model training and comparison. |
| [heart_disease_dignosis_openfe.ipynb](heart_disease_dignosis_openfe.ipynb) | OpenFE tuning, dataset export, and three-way comparison. |
| [evaluate_val.py](evaluate_val.py), [evaluate_test.py](evaluate_test.py) | Reusable CV, training, and evaluation. |
| [stacking_ensemble.py](stacking_ensemble.py) | Build and fit the training-only stacking model. |
| [openfe_features.py](openfe_features.py) | Automated discovery, boosting comparison, and forward selection. |
| [splits/](splits/) | Baseline `train/val/test.csv`, `fe_train/val/test.csv`, `openfe_train/val/test.csv`; formulas, configuration, and row indices in `openfe_feature_info.json`. |

Run from the project directory using the same Python environment in the terminal and Jupyter kernel:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel
python -m pip install -r requirements_openfe.txt
jupyter notebook
```

Use **Run All** in this order: `feature_engineering.ipynb` → `heart_disease_diagnosis.ipynb` → `heart_disease_dignosis_openfe.ipynb`. The OpenFE notebook regenerates its three CSVs.

Seed 42 is used for `PYTHONHASHSEED`, NumPy, `random`, and supported split/model parameters. Tested environment: Python 3.14.7, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, OpenFE 0.0.12, and LightGBM 4.7.0. Library versions and feature ordering can affect the results.

## 5. Limitations and lessons

- The test set has only 31 samples: one prediction changes accuracy by about 3.23 percentage points. High validation accuracy should not be presented as test performance.
- Preprocessing and feature discovery/selection fit on the outer training set before CV folds. CV scores are tuning estimates and may be optimistic; nested CV with the full pipeline refitted inside each fold is a future improvement.
- To reproduce the solution, manual `age_bin` uses `pd.cut` separately on each split, so bin edges can differ. A subsequent implementation should learn the edges on train and reuse them.
- This OpenFE experiment searches first-order numeric transformations, not every configuration supported by the library.
- OpenFE configuration selection uses train; validation/test are reported afterwards. However, test results have been inspected across development iterations, so a fresh holdout is needed for a final independent assessment.

## 6. Sources and acknowledgments

The baseline/manual-FE workflow reproduces an AIO2025 course solution. Extensions include reusable evaluation modules, a separate stacking implementation, OpenFE tuning, and analysis of metric differences. The project brief is available in [references/](references/).

- [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — dataset and attribute codes; UCI distributes the dataset under CC BY 4.0.
- [Dataset DOI: 10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
- [OpenFE source and documentation](https://github.com/IIIS-Li-Group/OpenFE).
- [OpenFE Parameters Tuning](https://openfe-document.readthedocs.io/en/latest/parameter_tuning.html).
