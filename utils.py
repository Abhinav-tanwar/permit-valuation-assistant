from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def reg(actual, pred):
    return dict(
        rmse=float(np.sqrt(mean_squared_error(actual, pred))),
        mae=float(mean_absolute_error(actual, pred)),
        r_squared=float(r2_score(actual, pred)),
    )


def clf(actual, pred, prob=None):
    r = dict(
        accuracy=float(accuracy_score(actual, pred)),
        precision=float(precision_score(actual, pred, zero_division=0)),
        recall=float(recall_score(actual, pred, zero_division=0)),
        f1=float(f1_score(actual, pred, zero_division=0)),
        confusion=confusion_matrix(actual, pred, labels=[0, 1]).tolist(),
    )
    if prob is not None:
        r.update(
            roc_auc=float(roc_auc_score(actual, prob)),
            average_precision=float(average_precision_score(actual, prob)),
        )
    return r


def save(result):
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    return result
