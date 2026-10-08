import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def get_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted probability for the positive class.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score, and ROC-AUC.
    """

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


def evaluate_model(model, X_test, y_test):
    """
    Generate predictions and calculate evaluation metrics
    for a trained classification model.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = get_metrics(y_test, y_pred, y_prob)

    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix for binary classification.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob, label="Model"):
    """
    Plot the ROC curve for a binary classification model.

    Returns
    -------
    fpr : array
        False positive rates.
    tpr : array
        True positive rates.
    auc_score : float
        ROC-AUC score.
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{label} (AUC = {auc_score:.3f})"
    )

    return fpr, tpr, auc_score
