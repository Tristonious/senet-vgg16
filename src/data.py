import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE


def find_imagenette() -> str:
    """
    Locates the Imagenette dataset root directory.

    Checks (in order):
      1. IMAGENETTE_PATH environment variable
      2. A set of common local paths
      3. Folders named imagenette* placed next to this file or in the project root

    Returns:
        Path to the dataset root (must contain 'train/' and 'val/' subdirs).

    Raises:
        FileNotFoundError with download instructions if the dataset is not found.
    """
    candidates = [
        os.environ.get("IMAGENETTE_PATH", ""),
        os.path.expanduser("~/data/imagenette2-320"),
        os.path.expanduser("~/imagenette2-320"),
    ]
    proj_dir = os.path.dirname(os.path.dirname(__file__))
    for name in ["imagenette2", "imagenette2-320", "imagenette2-160", "imagenette"]:
        candidates.append(os.path.join(proj_dir, name))

    for path in dict.fromkeys(c for c in candidates if c):
        if os.path.isdir(path) and \
           os.path.isdir(os.path.join(path, "train")) and \
           os.path.isdir(os.path.join(path, "val")):
            print(f"[INFO] Found Imagenette at: {path}")
            return path

    raise FileNotFoundError(
        "Imagenette dataset not found.\n\n"
        "Download instructions:\n"
        "  wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz\n"
        "  tar -xf imagenette2-320.tgz\n\n"
        "Then either:\n"
        "  (a) Place the extracted folder next to this repo, or\n"
        "  (b) Set the IMAGENETTE_PATH environment variable:\n"
        "        export IMAGENETTE_PATH=/path/to/imagenette2-320\n"
    )


def build_datasets(
    root: str,
    batch_size: int = 32,
    max_train: int = None,
    max_val: int = None,
):
    """
    Builds tf.data pipelines from an Imagenette directory tree.

    Expected structure:
        root/train/{class}/image.jpg
        root/val/{class}/image.jpg

    Args:
        root (str): Imagenette root directory returned by find_imagenette().
        batch_size (int): Mini-batch size.
        max_train (int | None): Cap on training samples (useful for quick runs).
        max_val (int | None): Cap on validation samples.

    Returns:
        train_ds, val_ds: Preprocessed, prefetched tf.data.Dataset objects.
    """
    def _load(directory, shuffle):
        return tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            shuffle=shuffle,
            seed=1337,
        )

    train_ds = _load(os.path.join(root, "train"), shuffle=True)
    val_ds   = _load(os.path.join(root, "val"),   shuffle=False)

    if max_train is not None:
        train_ds = train_ds.unbatch().take(max_train).batch(batch_size)
    if max_val is not None:
        val_ds = val_ds.unbatch().take(max_val).batch(batch_size)

    def _prep(x, y):
        return preprocess_input(tf.cast(x, tf.float32)), y

    train_ds = train_ds.map(_prep, num_parallel_calls=AUTOTUNE).cache().repeat().prefetch(AUTOTUNE)
    val_ds   = val_ds.map(_prep,   num_parallel_calls=AUTOTUNE).cache().repeat().prefetch(AUTOTUNE)
    return train_ds, val_ds
