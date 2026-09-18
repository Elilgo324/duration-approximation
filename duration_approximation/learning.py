"""Three fixed model families, selected only by explicit validation MAE."""
from __future__ import annotations

import copy
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .runtime import save_json


def metrics(truth, prediction):
    error = np.asarray(prediction)-np.asarray(truth)
    valid = np.isfinite(error)
    if not valid.any():
        return dict(count=len(error), coverage=0.0, mae_seconds=None, p95_absolute_seconds=None, bias_seconds=None)
    error = error[valid]
    return dict(count=len(valid), coverage=float(valid.mean()), mae_seconds=float(np.abs(error).mean()),
                p95_absolute_seconds=float(np.quantile(np.abs(error), .95)), bias_seconds=float(error.mean()))


def prepare_features(x):
    return np.where(np.isfinite(x), x, np.nan)


def train_models(x, y, splits, config, seed, output, smoke=False):
    train, validation, test = (np.flatnonzero(splits == name) for name in ("train", "validation", "test"))
    x = prepare_features(x)
    rows, fitted = [], {}
    for name in ("ridge", "boosted_trees", "mlp"):
        parameters = config[name]
        candidates = parameters["max_leaf_nodes"] if name == "boosted_trees" else parameters["alpha"]
        best = None
        best_mae = np.inf
        history = []
        started = time.perf_counter()
        for value in candidates:
            if name == "ridge":
                estimator = Ridge(alpha=value)
            elif name == "boosted_trees":
                estimator = HistGradientBoostingRegressor(max_leaf_nodes=value, max_iter=20 if smoke else parameters["max_iter"],
                    learning_rate=parameters["learning_rate"], l2_regularization=parameters["l2_regularization"],
                    early_stopping=False, random_state=seed)
            else:
                estimator = MLPRegressor(hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]), alpha=value,
                    learning_rate_init=parameters["learning_rate_init"], batch_size=min(256,len(train)),
                    random_state=seed, max_iter=1, early_stopping=False)
            transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
            xt = transform.fit_transform(x[train])
            xv = transform.transform(x[validation])
            # Target scaling is fitted only on training data and shared by all families.
            mean, scale = float(y[train].mean()), max(float(y[train].std()), 1.0)
            yt = (y[train]-mean)/scale
            if name == "mlp":
                candidate_best, candidate_score, stale = None, np.inf, 0
                for epoch in range(8 if smoke else parameters["max_epochs"]):
                    estimator.partial_fit(xt, yt)
                    prediction = np.maximum(0, estimator.predict(xv)*scale+mean)
                    score = float(np.abs(prediction-y[validation]).mean())
                    history.append(dict(candidate=value, epoch=epoch+1, validation_mae_seconds=score))
                    if score < candidate_score:
                        candidate_best, candidate_score, stale = copy.deepcopy(estimator), score, 0
                    else:
                        stale += 1
                    if stale >= parameters["patience"]:
                        break
                estimator, score = candidate_best, candidate_score
            else:
                estimator.fit(xt, yt)
                prediction = np.maximum(0, estimator.predict(xv)*scale+mean)
                score = float(np.abs(prediction-y[validation]).mean())
                history.append(dict(candidate=value, validation_mae_seconds=score))
            if score < best_mae:
                best_mae = score
                best = dict(transform=transform, estimator=estimator, target_mean=mean, target_scale=scale, selected=value)
        elapsed = time.perf_counter()-started
        model_path = output / f"{name}.joblib"
        joblib.dump(best, model_path)
        save_json(output / f"{name}_selection.json", dict(history=history, selected=best["selected"], validation_mae_seconds=best_mae))
        prediction = predict(best, x[test])
        np.save(output / f"{name}_test_predictions.npy", prediction)
        row = dict(method=name, **metrics(y[test], prediction), training_seconds=elapsed,
                   model_bytes=model_path.stat().st_size, validation_mae_seconds=best_mae,
                   train_mae_seconds=metrics(y[train], predict(best,x[train]))["mae_seconds"])
        rows.append(row)
        fitted[name] = best
        print(f"Model {name}: held-out MAE {row['mae_seconds']:.2f}s; trained in {elapsed:.1f}s", flush=True)
    return rows, fitted


def predict(model, x):
    x = model["transform"].transform(prepare_features(x))
    return np.maximum(0, model["estimator"].predict(x)*model["target_scale"]+model["target_mean"])
