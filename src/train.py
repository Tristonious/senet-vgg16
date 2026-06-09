import os
import json
import tensorflow as tf


def train_and_eval(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    tag: str,
    output_dir: str,
    epochs: int = 2,
    steps_per_epoch: int = None,
    validation_steps: int = None,
):
    """
    Trains a model and evaluates it on the validation set.

    Saves:
        {output_dir}/{tag}_best.keras   — best checkpoint by val_accuracy
        {output_dir}/{tag}_history.json — per-epoch training history
        {output_dir}/{tag}_metrics.json — final evaluation metrics

    Args:
        model: Compiled tf.keras.Model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        tag (str): Identifier used in filenames (e.g. "baseline", "se_before_conv1_1").
        output_dir (str): Directory for saving checkpoints and metrics.
        epochs (int): Number of training epochs.
        steps_per_epoch (int | None): Steps per epoch; inferred from dataset if None.
        validation_steps (int | None): Validation steps; inferred if None.

    Returns:
        dict: Final evaluation metrics from model.evaluate().
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"{tag}_best.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    eval_res = model.evaluate(val_ds, steps=validation_steps, return_dict=True)

    with open(os.path.join(output_dir, f"{tag}_history.json"), "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    with open(os.path.join(output_dir, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in eval_res.items()}, f, indent=2)

    print(f"[RESULT] {tag}: {eval_res}")
    return eval_res
