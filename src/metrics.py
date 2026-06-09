import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


def fm_metrics(act: np.ndarray):
    """
    Computes spatial attention quality metrics for a single feature map.

    Args:
        act (np.ndarray): Shape (H, W, C) — a single image's feature map.

    Returns:
        entropy_mean (float): Mean per-channel Shannon entropy over spatial positions.
            Lower entropy = more concentrated/selective activations.
        focus10_mean (float): Mean fraction of activation mass in the top 10%
            of spatial positions per channel. Higher = more localized.
    """
    x = act.copy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    flat = x.reshape(-1, x.shape[-1])          # (H*W, C)
    sums = flat.sum(axis=0) + 1e-8
    p = flat / (sums + 1e-12)
    entropy_mean = float(-(p * np.log2(p + 1e-12)).sum(axis=0).mean())

    k = max(1, int(0.10 * flat.shape[0]))
    focus10_mean = float((np.sort(flat, axis=0)[-k:, :].sum(axis=0) / sums).mean())
    return entropy_mean, focus10_mean


def compare_feature_map_metrics(
    baseline_model: tf.keras.Model,
    se_model: tf.keras.Model,
    val_ds,
    layer_name: str,
    out_json: str,
    out_plot: str,
    take_batches: int = 5,
):
    """
    Compares feature-map quality metrics (entropy and top-10% focus) between
    a baseline model and an SE-augmented model at a specified layer.

    Saves:
        out_json — JSON summary of mean metrics for both models.
        out_plot — Grouped bar chart comparing baseline vs SE metrics.

    Args:
        baseline_model: Baseline tf.keras.Model.
        se_model: SE-augmented tf.keras.Model.
        val_ds: Validation tf.data.Dataset.
        layer_name (str): Name of the layer to probe (must exist in both models).
        out_json (str): Output path for the JSON metrics summary.
        out_plot (str): Output path for the comparison bar chart.
        take_batches (int): Number of validation batches to sample.
    """
    def _sample(model):
        try:
            probe = tf.keras.Model(model.input, model.get_layer(layer_name).output)
        except Exception as e:
            print(f"[WARN] Could not probe {layer_name}: {e}")
            return None
        ents, focs = [], []
        for xb, _ in val_ds.take(take_batches):
            for a in probe.predict(xb, verbose=0):
                e, f = fm_metrics(a)
                ents.append(e)
                focs.append(f)
        if not ents:
            return None
        return {"entropy_mean": float(np.mean(ents)), "focus10_mean": float(np.mean(focs))}

    base_stats = _sample(baseline_model)
    se_stats   = _sample(se_model)
    summary = {"layer": layer_name, "baseline": base_stats, "se_model": se_stats}

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Feature-map metrics saved to {out_json}")

    labels = ["entropy_mean", "focus10_mean"]
    bv = [base_stats.get(k, 0.0) if base_stats else 0.0 for k in labels]
    sv = [se_stats.get(k,   0.0) if se_stats   else 0.0 for k in labels]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, bv, width, label="baseline")
    plt.bar(x + width / 2, sv, width, label="with SE")
    plt.xticks(x, ["Entropy", "Top-10% focus"])
    plt.ylabel("metric value")
    plt.title(f"Feature-map metrics at {layer_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"[INFO] Feature-map comparison plot saved to {out_plot}")
