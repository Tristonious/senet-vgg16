import os
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = (160, 160)


def visualize_feature_maps(
    model: tf.keras.Model,
    image_paths,
    layer_names,
    tag: str,
    output_dir: str,
    max_channels: int = 16,
):
    """
    Saves feature-map montage images for each (image, layer) combination.

    For each image, builds a sub-model that outputs activations at the
    specified layers, then saves a grid of up to max_channels channel maps.

    Args:
        model: tf.keras.Model to probe.
        image_paths (list[str]): Paths to input images.
        layer_names (list[str]): Layer names to extract activations from.
        tag (str): Label used in output filenames and figure titles.
        output_dir (str): Directory for saved images.
        max_channels (int): Maximum number of channels to display per layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = [model.get_layer(name).output for name in layer_names]
    act_model = tf.keras.Model(inputs=model.input, outputs=outputs)

    for img_path in image_paths:
        img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
        x = tf.keras.utils.img_to_array(img)[None, ...]
        x = preprocess_input(x)
        activations = act_model.predict(x, verbose=0)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for lname, act in zip(layer_names, activations):
            act = act[0]
            show_c = min(act.shape[-1], max_channels)
            cols = int(math.sqrt(show_c))
            rows = math.ceil(show_c / cols)

            fig = plt.figure(figsize=(cols * 2, rows * 2))
            for i in range(show_c):
                ax = fig.add_subplot(rows, cols, i + 1)
                fm = act[:, :, i]
                fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                ax.imshow(fm, cmap="viridis")
                ax.axis("off")
            fig.suptitle(f"{tag} — {lname} — {base_name}", y=0.95)
            out_path = os.path.join(output_dir, f"{tag}_{lname}_{base_name}.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    print(f"[INFO] Feature map visualizations saved to {output_dir}")


def save_before_after_bar(output_dir: str):
    """
    Reads per-position *_history.json files from output_dir and produces
    a grouped bar chart comparing baseline vs SE-augmented validation accuracy
    at each of the six insertion positions.
    """
    positions = [
        "before_conv1_1",
        "between_pool1_conv2_1",
        "between_pool2_conv3_1",
        "between_pool3_conv4_1",
        "between_pool4_conv5_1",
        "between_pool5_dense",
    ]

    def _load_final_val(path):
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        val_acc = j.get("val_accuracy") or j.get("val_acc")
        return float(val_acc[-1]) if val_acc else None

    baseline_val = _load_final_val(os.path.join(output_dir, "baseline_history.json"))
    se_vals = [
        _load_final_val(os.path.join(output_dir, f"se_{pos}_history.json")) or 0.0
        for pos in positions
    ]

    x = np.arange(len(positions))
    width = 0.35
    baseline_list = [baseline_val] * len(positions)

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, baseline_list, width, label="baseline (before)", color="tab:blue", alpha=0.9)
    plt.bar(x + width / 2, se_vals,       width, label="with SE (after)",   color="tab:orange", alpha=0.9)

    for i, (b, s) in enumerate(zip(baseline_list, se_vals)):
        plt.text(i - width / 2, b + 0.002, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
        plt.text(i + width / 2, s + 0.002, f"{s:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, positions, rotation=30, ha="right")
    plt.ylim(0, max(max(baseline_list), max(se_vals)) + 0.05)
    plt.ylabel("Final Validation Accuracy")
    plt.title("Before (baseline) vs After (SE inserted) — Final Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "before_after_grouped_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Comparison bar chart saved to {out_path}")


def save_se_excitation_plots(
    model: tf.keras.Model,
    val_ds,
    se_fc2_layer_name: str,
    out_prefix: str,
    take_batches: int = 5,
):
    """
    Saves a histogram and violin plot of SE excitation gate values
    (output of the sigmoid FC2 layer) sampled from the validation set.

    Args:
        model: tf.keras.Model containing the SE fc2 layer.
        val_ds: Validation tf.data.Dataset.
        se_fc2_layer_name (str): Name of the SE sigmoid Dense layer.
        out_prefix (str): File path prefix; _hist.png and _violin.png are appended.
        take_batches (int): Number of batches to sample.
    """
    try:
        probe = tf.keras.Model(model.input, model.get_layer(se_fc2_layer_name).output)
    except Exception as e:
        print(f"[WARN] Could not probe {se_fc2_layer_name}: {e}")
        return

    batches = [probe.predict(xb, verbose=0) for xb, _ in val_ds.take(take_batches)]
    if not batches:
        return
    W = np.vstack(batches)

    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(W.flatten(), bins=60)
    plt.title(f"SE excitation weights — {se_fc2_layer_name}")
    plt.xlabel("weight")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_prefix + "_hist.png")
    plt.close()

    # Violin (top 24 channels by mean)
    order = np.argsort(W.mean(axis=0))[::-1]
    topk = min(24, W.shape[1])
    plt.figure(figsize=(12, 5))
    plt.violinplot([W[:, int(c)] for c in order[:topk]], showmeans=True)
    plt.title(f"SE excitation per-channel (top {topk}) — {se_fc2_layer_name}")
    plt.xlabel("channel (sorted by mean)")
    plt.ylabel("weight")
    plt.tight_layout()
    plt.savefig(out_prefix + "_violin.png")
    plt.close()
    print(f"[INFO] SE excitation plots saved to {out_prefix}_hist.png / _violin.png")
