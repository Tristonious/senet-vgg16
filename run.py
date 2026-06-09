import os
import json
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src import (
    find_imagenette,
    build_datasets,
    build_baseline_model,
    build_model_with_se,
    SE_POSITIONS,
    train_and_eval,
    visualize_feature_maps,
    save_before_after_bar,
    save_se_excitation_plots,
    compare_feature_map_metrics,
)

# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 2
MAX_TRAIN    = 500     # set to None to use the full training set
MAX_VAL      = 250     # set to None to use the full validation set
OUTPUT_DIR   = "results"
FIGURES_DIR  = "figures"
SE_RATIO     = 16

# Layers to visualize feature maps at
VIS_LAYERS = ["block1_pool", "block2_pool", "block3_pool", "block4_pool", "block5_pool"]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    root = find_imagenette()
    train_ds, val_ds = build_datasets(root, batch_size=BATCH_SIZE,
                                      max_train=MAX_TRAIN, max_val=MAX_VAL)

    steps_per_epoch  = (MAX_TRAIN // BATCH_SIZE) if MAX_TRAIN else None
    validation_steps = (MAX_VAL   // BATCH_SIZE) if MAX_VAL   else None

    # ── Baseline ────────────────────────────────────────────────────────────
    baseline = build_baseline_model()
    baseline_res = train_and_eval(
        baseline, train_ds, val_ds, tag="baseline",
        output_dir=OUTPUT_DIR, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
    )

    # ── SE positions ────────────────────────────────────────────────────────
    all_results = {"baseline": baseline_res}
    for pos in SE_POSITIONS:
        print(f"\n[RUN] SE insertion: {pos}")
        model = build_model_with_se(position=pos, se_ratio=SE_RATIO)
        res = train_and_eval(
            model, train_ds, val_ds, tag=f"se_{pos}",
            output_dir=OUTPUT_DIR, epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        )
        all_results[pos] = res

    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ── Comparison bar chart ─────────────────────────────────────────────────
    save_before_after_bar(OUTPUT_DIR)

    # ── Feature map visualization (1 sample image) ──────────────────────────
    val_dir = os.path.join(root, "val")
    sample_paths = []
    for cls in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_paths.append(os.path.join(cls_dir, fname))
    random.shuffle(sample_paths)
    sample_paths = sample_paths[:1]

    visualize_feature_maps(baseline, sample_paths, VIS_LAYERS, tag="baseline",
                           output_dir=FIGURES_DIR)

    se_pool3 = build_model_with_se("between_pool3_conv4_1")
    visualize_feature_maps(se_pool3, sample_paths, VIS_LAYERS, tag="se_after_pool3",
                           output_dir=FIGURES_DIR)

    # ── SE excitation diagnostics ────────────────────────────────────────────
    save_se_excitation_plots(
        se_pool3, val_ds,
        se_fc2_layer_name="se_after_pool3_fc2",
        out_prefix=os.path.join(FIGURES_DIR, "se_after_pool3_excitation"),
        take_batches=5,
    )

    # ── Feature-map quality metrics (baseline vs SE at block3_pool) ──────────
    compare_feature_map_metrics(
        baseline, se_pool3, val_ds,
        layer_name="block3_pool",
        out_json=os.path.join(OUTPUT_DIR, "fm_metrics_block3.json"),
        out_plot=os.path.join(FIGURES_DIR, "fm_metrics_block3.png"),
    )

    print(f"\n[DONE] Results in {OUTPUT_DIR}/  |  Figures in {FIGURES_DIR}/")
