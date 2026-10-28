"""Generate train-only OpenFE features and export the existing 80:10:10 splits.

Uses first-order row-wise numeric operators and train-only parameter selection.
Compare feature boosting on/off, remove constants/duplicates, and perform
forward feature selection without a fixed feature-count cap.
"""

import json
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from openfe import OpenFE, get_candidate_features, tree_to_formula
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "target"]
ROW_WISE_OPERATORS = {"abs", "log", "sqrt", "square", "sigmoid", "round", "residual",
                      "min", "max", "+", "-", "*", "/"}


class _QuietOpenFE(OpenFE):
    """Keep LightGBM worker logs compact without changing its calculations."""

    def _evaluate(self, *args, **kwargs):
        log = io.StringIO()
        try:
            with warnings.catch_warnings(), redirect_stdout(log):
                warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated.*")
                return super()._evaluate(*args, **kwargs)
        except BaseException:
            print(log.getvalue())
            raise

    def get_init_score(self, *args, **kwargs):
        log = io.StringIO()
        try:
            with warnings.catch_warnings(), redirect_stdout(log):
                warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated.*")
                return super().get_init_score(*args, **kwargs)
        except BaseException:
            print(log.getvalue())
            raise


def _augment(frame, nodes):
    result = frame.copy()
    for index, node in enumerate(nodes):
        # These operators depend only on this row, not other split statistics.
        result[f"openfe_{index:03d}"] = node.calculate(frame).replace([np.inf, -np.inf], np.nan)
        node.delete()
    return result


def _forward_select(X, y, seed, initial_features):
    """Keep original features and add generated ones if training CV improves.

    Scores are reused tuning estimates, not unbiased estimates of the full
    discovery/selection procedure. Val/test are never accessed here.
    """
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y))
    selected = list(initial_features)
    remaining = [column for column in X.columns if column not in selected]
    best_score = float(cross_val_score(
        KNeighborsClassifier(n_neighbors=5), X[selected], y,
        cv=folds, scoring="accuracy", n_jobs=1,
    ).mean())
    history = [{"step": 0, "added_feature": "Original features", "cv_accuracy": best_score,
                "feature_count": len(selected)}]
    while remaining:
        candidate, candidate_score = None, best_score
        for column in remaining:
            score = cross_val_score(
                KNeighborsClassifier(n_neighbors=5), X[selected + [column]], y,
                cv=folds, scoring="accuracy", n_jobs=1,
            ).mean()
            if score > candidate_score + 1e-12:
                candidate, candidate_score = column, float(score)
        if candidate is None:
            break
        selected.append(candidate)
        remaining.remove(candidate)
        best_score = candidate_score
        history.append({"step": len(history), "added_feature": candidate,
                        "cv_accuracy": best_score, "feature_count": len(selected)})
    return selected, best_score, history


def build_openfe_splits(project_dir, seed=42):
    """Compare feature boosting on/off and select features using train only."""
    project_dir = Path(project_dir).resolve()
    raw = pd.read_csv(project_dir / "data" / "cleveland.csv", header=None, names=COLUMNS)
    X = raw.drop(columns="target").apply(pd.to_numeric, errors="coerce")
    y = (pd.to_numeric(raw["target"], errors="raise") > 0).astype(int)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=seed
    )
    trials = []
    for feature_boosting in [False, True]:
        print(f"Tuning OpenFE: feature_boosting={feature_boosting}", flush=True)
        candidates = get_candidate_features(numerical_features=NUMERIC_COLS, order=1)
        candidates = [node for node in candidates if node.name in ROW_WISE_OPERATORS]
        discovery = _QuietOpenFE()
        with TemporaryDirectory(prefix="heart_openfe_tuning_") as temp_dir:
            discovery.fit(
                data=X_train.copy(), label=y_train.to_frame().copy(),
                task="classification", categorical_features=CATEGORICAL_COLS,
                candidate_features_list=candidates,
                metric="binary_logloss", n_data_blocks=1,
                min_candidate_features=len(candidates), feature_boosting=feature_boosting,
                stage1_metric="predictive", stage2_metric="gain_importance",
                stage2_params={
                    "n_estimators": 1000, "num_leaves": 16, "importance_type": "gain",
                    "random_state": seed, "n_jobs": 1, "verbosity": -1,
                    "deterministic": True, "force_col_wise": True,
                },
                tmp_save_path=str(Path(temp_dir) / "discovery.feather"),
                n_jobs=1, seed=seed, verbose=False,
            )
        generated = list(discovery.new_features_scores_list)
        nodes = [node for node, _ in generated]
        formulas = {f"openfe_{i:03d}": tree_to_formula(node) for i, node in enumerate(nodes)}
        augmented_train = _augment(X_train, nodes)
        numeric_features = NUMERIC_COLS + list(formulas)
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_COLS),
        ], verbose_feature_names_out=False).set_output(transform="pandas")
        variance_filter = VarianceThreshold().set_output(transform="pandas")
        train_encoded = variance_filter.fit_transform(preprocessor.fit_transform(augmented_train))
        unique_columns, duplicate_columns = [], []
        for column in train_encoded.columns:
            duplicate_of = next((old for old in unique_columns if np.allclose(
                train_encoded[column], train_encoded[old], atol=1e-12, rtol=1e-12
            )), None)
            if duplicate_of is None:
                unique_columns.append(column)
            else:
                duplicate_columns.append({"feature": column, "duplicate_of": duplicate_of})
        train_encoded = train_encoded[unique_columns]
        original_features = [column for column in unique_columns if column not in formulas]
        selected, cv_score, selection_history = _forward_select(
            train_encoded, y_train, seed, initial_features=original_features
        )
        if not selected:
            raise ValueError("Forward selection did not select any feature.")
        print(f"  Retained {len(selected)} of {len(unique_columns)} unique features; "
              f"training CV accuracy={cv_score:.4f}", flush=True)
        trials.append({
            "feature_boosting": feature_boosting, "candidate_count": len(candidates),
            "generated": generated, "nodes": nodes, "formulas": formulas,
            "numeric_features": numeric_features, "preprocessor": preprocessor,
            "variance_filter": variance_filter, "unique_columns": unique_columns,
            "duplicate_columns": duplicate_columns, "train_encoded": train_encoded,
            "selected": selected, "cv_score": cv_score, "selection_history": selection_history,
        })
    # Highest training CV score, then fewer features; no external val/test access.
    winner = min(trials, key=lambda trial: (-trial["cv_score"], len(trial["selected"]), trial["feature_boosting"]))
    selected_features = winner["selected"]
    encoded = {"train": winner["train_encoded"]}
    for split, frame in [("val", X_val), ("test", X_test)]:
        augmented = _augment(frame, winner["nodes"])
        encoded[split] = winner["variance_filter"].transform(
            winner["preprocessor"].transform(augmented)
        )[winner["unique_columns"]]
    discrete = np.array([column not in winner["numeric_features"] for column in encoded["train"].columns])
    mi = mutual_info_classif(encoded["train"].values, y_train.values,
                            discrete_features=discrete, random_state=seed)
    ranking = pd.Series(mi, index=encoded["train"].columns).sort_values(ascending=False)
    ranking = ranking.rename_axis("feature").reset_index(name="mutual_information")
    ranking["formula"] = ranking["feature"].map(winner["formulas"]).fillna("")
    ranking["selected_by_cv"] = ranking["feature"].isin(selected_features)
    labels = {"train": y_train, "val": y_val, "test": y_test}
    split_dir = project_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    exports = {}
    for split in ["train", "val", "test"]:
        frame = encoded[split][selected_features].assign(target=labels[split])
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Non-finite values in openfe_{split}.")
        baseline = pd.read_csv(split_dir / f"{split}.csv")
        if not np.array_equal(frame["target"].to_numpy(), baseline["target"].to_numpy()):
            raise ValueError(f"Split labels do not align with baseline {split}.")
        frame.to_csv(split_dir / f"openfe_{split}.csv", index=False)
        exports[split] = frame
    comparison = [{
        "feature_boosting": trial["feature_boosting"], "n_data_blocks": 1,
        "min_candidate_features": trial["candidate_count"],
        "generated_features": len(trial["generated"]),
        "duplicate_features_removed": len(trial["duplicate_columns"]),
        "selected_features": len(trial["selected"]), "train_cv_accuracy": trial["cv_score"],
        "selected_config": trial is winner,
    } for trial in trials]
    metadata = {
        "seed": seed, "feature_limit": None, "feature_count": len(selected_features),
        "generated_feature_count": len(winner["generated"]), "candidate_count": winner["candidate_count"],
        "max_order": 1, "operator_policy": "row-wise numeric transformations",
        "stage1_metric": "predictive", "stage2_metric": "gain_importance",
        "n_data_blocks": 1, "min_candidate_features": winner["candidate_count"],
        "feature_boosting": winner["feature_boosting"], "tuning_comparison": comparison,
        "selection_method": "keep original features; forward-select generated features until no increase in train-only 5-fold KNN accuracy",
        "selection_cv_accuracy": winner["cv_score"], "selection_history": winner["selection_history"],
        "selection_cv_note": "Tuning estimates reuse discovery/preprocessing fitted on outer train; not unbiased nested CV.",
        "duplicate_features_removed": winner["duplicate_columns"],
        "generated_formulas": winner["formulas"], "exported_features": selected_features,
        "training_row_indices": X_train.index.tolist(),
        "validation_row_indices": X_val.index.tolist(),
        "test_row_indices": X_test.index.tolist(),
    }
    (split_dir / "openfe_feature_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generated_ranking = pd.DataFrame([
        {"feature": f"openfe_{index:03d}", "formula": tree_to_formula(node),
         "openfe_gain": float(score), "selected_by_cv": f"openfe_{index:03d}" in selected_features}
        for index, (node, score) in enumerate(winner["generated"])
    ])
    return exports, ranking, generated_ranking, metadata
