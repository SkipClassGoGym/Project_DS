"""Reusable model training, stratified CV, and validation metrics."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class KMeansClassifier(ClassifierMixin, BaseEstimator):
    """Map each cluster to its majority training class; never use evaluation labels."""

    def __init__(self, random_state=42, n_init="auto", init="random"):
        self.random_state = random_state
        self.n_init = n_init
        self.init = init

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_features_in_ = X.shape[1]
        self.kmeans_ = KMeans(
            n_clusters=len(self.classes_),
            random_state=self.random_state,
            n_init=self.n_init,
            init=self.init,
        )
        clusters = self.kmeans_.fit_predict(X)
        fallback = self.classes_[np.argmax(counts)]
        self.cluster_labels_ = {}
        for cluster in range(len(self.classes_)):
            labels, label_counts = np.unique(y[clusters == cluster], return_counts=True)
            self.cluster_labels_[cluster] = (
                labels[np.argmax(label_counts)] if len(labels) else fallback
            )
        return self

    def predict(self, X):
        check_is_fitted(self, ["kmeans_", "cluster_labels_"])
        return np.asarray(
            [self.cluster_labels_[cluster] for cluster in self.kmeans_.predict(X)]
        )


class HardVotingEnsemble:
    """Equal votes from fitted classifiers; use a train-chosen class for ties."""

    def __init__(self, models, tie_break_class):
        if not models:
            raise ValueError("At least one fitted model is required.")
        self.models = models
        self.tie_break_class = tie_break_class
        self.classes_ = np.unique(np.concatenate([model.classes_ for model in models.values()]))
        if tie_break_class not in self.classes_:
            raise ValueError("The tie-break class must be a known training class.")

    def predict(self, X):
        votes = np.column_stack([model.predict(X) for model in self.models.values()])
        counts = np.column_stack([(votes == label).sum(axis=1) for label in self.classes_])
        winners = self.classes_[np.argmax(counts, axis=1)]
        tie_index = np.flatnonzero(self.classes_ == self.tie_break_class)[0]
        tied = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1
        tie_class_is_winner = counts[:, tie_index] == counts.max(axis=1)
        winners[tied & tie_class_is_winner] = self.tie_break_class
        return winners


def _search(estimator, parameter, candidates, X_train, y_train, n_splits, random_state):
    min_class_count = int(pd.Series(y_train).value_counts().min())
    if min_class_count < n_splits:
        raise ValueError(f"Each training class must have at least {n_splits} rows.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator,
        {parameter: list(candidates)},
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        refit="accuracy",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame(
        {
            parameter: results[f"param_{parameter}"].tolist(),
            "mean_f1_macro": results["mean_test_f1_macro"],
            "std_f1_macro": results["std_test_f1_macro"],
            "mean_accuracy": results["mean_test_accuracy"],
            "std_accuracy": results["std_test_accuracy"],
            "rank": results["rank_test_accuracy"],
        }
    )
    return search.best_estimator_, search.best_params_[parameter], table


def find_optimal_k(X_train, y_train, k_values=None, n_splits=5, random_state=42):
    """Match the reference: search k=1..20 using stratified CV accuracy."""
    candidates = list(range(1, 21)) if k_values is None else list(k_values)
    smallest_fold_train = len(X_train) - int(np.ceil(len(X_train) / n_splits))
    candidates = [k for k in candidates if 1 <= k <= smallest_fold_train]
    if not candidates:
        raise ValueError("No valid neighbor counts for the training fold size.")
    return _search(
        KNeighborsClassifier(), "n_neighbors", candidates,
        X_train, y_train, n_splits, random_state,
    )


def find_optimal_depth(X_train, y_train, depth_values=None, n_splits=5, random_state=42):
    """Match the reference: search depths 2..5 using stratified CV accuracy."""
    candidates = list(range(2, 6)) if depth_values is None else list(depth_values)
    return _search(
        DecisionTreeClassifier(random_state=random_state), "max_depth", candidates,
        X_train, y_train, n_splits, random_state,
    )


def fit_models(X_train, y_train, n_splits=5, random_state=42):
    """Tune KNN/DT on train and fit all four models without val/test access."""
    knn, best_k, knn_cv = find_optimal_k(
        X_train, y_train, n_splits=n_splits, random_state=random_state
    )
    tree, best_depth, tree_cv = find_optimal_depth(
        X_train, y_train, n_splits=n_splits, random_state=random_state
    )
    models = {
        "KNN": knn,
        "Decision Tree": tree,
        "KMeans": KMeansClassifier(random_state=random_state).fit(X_train, y_train),
        "Gaussian Naive Bayes": GaussianNB().fit(X_train, y_train),
    }
    return models, {"k": best_k, "depth": best_depth}, {"KNN": knn_cv, "Decision Tree": tree_cv}


def evaluate_models(models, X, y, dataset, split):
    """Shared evaluation used by validation and test; does not fit models."""
    rows, confusion_matrices, predictions = [], {}, {}
    for name, model in models.items():
        prediction = model.predict(X)
        labels = model.classes_
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, prediction, labels=labels, average="macro", zero_division=0
        )
        rows.append(
            {
                "dataset": dataset, "split": split, "model": name,
                "accuracy": accuracy_score(y, prediction),
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
            }
        )
        confusion_matrices[name] = pd.DataFrame(
            confusion_matrix(y, prediction, labels=labels),
            index=pd.Index(labels, name="Actual"),
            columns=pd.Index(labels, name="Predicted"),
        )
        predictions[name] = np.asarray(prediction)
    return pd.DataFrame(rows), confusion_matrices, predictions


def evaluate_val(models, X_val, y_val, dataset="data"):
    """Evaluate fitted models on validation data."""
    return evaluate_models(models, X_val, y_val, dataset, "val")
