"""Stack KNN, Decision Tree, and Gaussian NB with a final KNN classifier."""

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def build_stacking_ensemble(
    knn_k=5, tree_depth=None, final_k=5, n_splits=5, random_state=42
):
    """Build an unfitted stack using out-of-fold probabilities as meta-features.

    For binary classification, each base model contributes its probability for
    class 1. The final KNN therefore learns from three probability columns.
    Original input features are not passed to the final classifier.
    """
    return StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=knn_k)),
            ("decision_tree", DecisionTreeClassifier(
                max_depth=tree_depth, random_state=random_state
            )),
            ("naive_bayes", GaussianNB()),
        ],
        final_estimator=KNeighborsClassifier(n_neighbors=final_k),
        stack_method="predict_proba",
        # Equivalent to the reference StackingClassifier default cv=5.
        cv=StratifiedKFold(n_splits=n_splits, shuffle=False),
        passthrough=False,
        n_jobs=1,
    )


def fit_stacking_ensemble(
    X_train, y_train, knn_k=5, tree_depth=None, final_k=5,
    n_splits=5, random_state=42,
):
    """Fit on train only; never supply validation or test data to this function.

    Base-model parameters can come from training-only CV. StackingClassifier
    generates out-of-fold predictions to train the final KNN, then refits the
    three base models on all training rows for future inference.
    """
    if n_splits < 2:
        raise ValueError("Stacking requires at least two cross-validation folds.")
    classes, counts = np.unique(np.asarray(y_train), return_counts=True)
    if len(classes) < 2 or counts.min() < n_splits:
        raise ValueError("Each training class must have at least n_splits rows.")
    smallest_fold_train = len(y_train) - int(np.ceil(len(y_train) / n_splits))
    if not 1 <= knn_k <= smallest_fold_train:
        raise ValueError("Base KNN k must fit within the smallest training fold.")
    if not 1 <= final_k <= len(y_train):
        raise ValueError("Final KNN k must fit within the training set.")
    model = build_stacking_ensemble(
        knn_k=knn_k, tree_depth=tree_depth, final_k=final_k,
        n_splits=n_splits, random_state=random_state,
    )
    return model.fit(X_train, y_train)
