import numpy as np
import torch

from .image_data_utils import calculate_image_data_metrics


__all__ = [
    "calculate_metrics",
    "print_metrics",
    "stopping_criterion",
]


def stopping_criterion(
    val_loss,
    metrics,
    best_metrics,
    epochs_no_improve,
):
    """
    Define stopping criterion for metrics from config['saving_metrics']
    best_metrics is updated only if every metric from best_metrics.keys() has improved

    :param val_loss: validation loss
    :param metrics: validation metrics
    :param best_metrics: the best metrics for the current epoch
    :param epochs_no_improve: number of epochs without best_metrics updating

    :return: epochs_no_improve, best_metrics
    """
    # get average metrics by class
    metrics = dict(metrics.mean(axis=1))
    # define condition best_metric >= metric for all except for loss
    metrics_mask = all(
        metrics[key] >= best_metrics[key] for key in best_metrics.keys() - {"loss"}
    )
    if not metrics_mask:
        epochs_no_improve += 1
        return epochs_no_improve, best_metrics
    if "loss" in list(best_metrics.keys()):
        if val_loss >= best_metrics["loss"]:
            epochs_no_improve += 1
            return epochs_no_improve, best_metrics
    # Updating best_metrics
    for key in list(best_metrics.keys()):
        if key == "loss":
            best_metrics[key] = val_loss
        else:
            best_metrics[key] = metrics[key]
    # Updating epochs_no_improve
    epochs_no_improve = 0
    return epochs_no_improve, best_metrics


def calculate_metrics(
    fin_targets,
    fin_outputs,
    prediction_threshold,
    verbose=False,
):
    # Get results
    softmax = torch.nn.Softmax(dim=1)
    results = softmax(torch.as_tensor(fin_outputs)).max(dim=1)[1]
    fin_targets = torch.as_tensor(fin_targets)
    # Calc metrics
    metrics = calculate_image_data_metrics(fin_targets, results, verbose)
    prediction_threshold = None
    return metrics, prediction_threshold


def check_metrics_names(metrics):
    allowed_metrics = [
        "loss",
        "Specificity",
        "Sensitivity",
        "G-mean",
        "f1-score",
        "fbeta2-score",
        "ROC-AUC",
        "AP",
        "Precision (PPV)",
        "NPV",
    ]

    assert all(
        [k in allowed_metrics for k in metrics.keys()]
    ), f"federated_params.server_saving_metrics can be only {allowed_metrics}, but get {metrics.keys()}"

    if list(metrics.keys()) != ["loss"]:
        not_loss_metrics = [k for k in metrics.keys() if k != "loss"]
        print(
            f"""Server best model updating based on {not_loss_metrics} metrics in ECG setup requires sending appropriate threshold to clients.\n
                At the current moment, this functionality are not supported.\n
                If you want to manage threshold, you can set up it in `training_params.prediction_threshold` config."""
        )
