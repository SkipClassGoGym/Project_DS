"""Test evaluation reusing the same metrics as validation."""

from evaluate_val import evaluate_models


def evaluate_test(models, X_test, y_test, dataset="data"):
    """Evaluate frozen models on test data; no fitting or parameter selection."""
    return evaluate_models(models, X_test, y_test, dataset, "test")
