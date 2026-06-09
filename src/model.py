import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

NUM_CLASSES = 10
IMAGE_SIZE = (160, 160)

SE_POSITIONS = [
    "before_conv1_1",
    "between_pool1_conv2_1",
    "between_pool2_conv3_1",
    "between_pool3_conv4_1",
    "between_pool4_conv5_1",
    "between_pool5_dense",
]


def se_block(input_tensor, ratio: int = 16, name: str = "se"):
    """
    Squeeze-and-Excitation block.

    Three steps:
      Squeeze  — GlobalAveragePooling2D collapses spatial dims to a channel descriptor.
      Excitation — two FC layers (Dense-ReLU-Dense-Sigmoid) produce per-channel gates.
      Scale    — channel-wise multiply recalibrates the input feature map.

    Args:
        input_tensor: 4-D tensor (batch, H, W, C).
        ratio (int): Reduction ratio for the bottleneck FC layer.
        name (str): Name prefix for all sub-layers.

    Returns:
        Recalibrated tensor with the same shape as input_tensor.
    """
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(input_tensor)
    se = layers.Dense(max(channels // ratio, 1), activation="relu",  name=f"{name}_fc1")(se)
    se = layers.Dense(channels,                  activation="sigmoid", name=f"{name}_fc2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([input_tensor, se])


def vgg16_backbone():
    """
    Loads VGG16 pretrained on ImageNet (include_top=False) and freezes
    all convolutional weights for use as a fixed feature extractor.
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,))
    for layer in base.layers:
        layer.trainable = False
    return base


def build_classifier_head(x, num_classes: int = NUM_CLASSES, name: str = "head"):
    """
    Lightweight classification head: GAP → Dense(256) → Dropout → Softmax.
    """
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dense(256, activation="relu", name=f"{name}_fc")(x)
    x = layers.Dropout(0.3, name=f"{name}_drop")(x)
    return layers.Dense(num_classes, activation="softmax", name=f"{name}_pred")(x)


def build_baseline_model() -> tf.keras.Model:
    """
    Baseline VGG16 + classification head, no SE block. Convolutional
    weights are frozen; only the head trains.
    """
    base = vgg16_backbone()
    outputs = build_classifier_head(base.output)
    model = models.Model(inputs=base.input, outputs=outputs, name="vgg16_baseline")
    for l in model.layers:
        if l.name.startswith("block"):
            l.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_with_se(position: str, se_ratio: int = 16) -> tf.keras.Model:
    """
    Inserts a Squeeze-and-Excitation block at one of six positions in VGG16.

    Positions:
        before_conv1_1        — between input and block1_conv1
        between_pool1_conv2_1 — after block1_pool, before block2_conv1
        between_pool2_conv3_1 — after block2_pool, before block3_conv1
        between_pool3_conv4_1 — after block3_pool, before block4_conv1
        between_pool4_conv5_1 — after block4_pool, before block5_conv1
        between_pool5_dense   — after block5_pool, before classifier head

    VGG16 conv weights are frozen; only the SE block and classifier head train.
    """
    if position not in SE_POSITIONS:
        raise ValueError(f"position must be one of {SE_POSITIONS}, got {position!r}")

    base = vgg16_backbone()
    inputs = base.input
    layer_dict = {l.name: l for l in base.layers}

    pool1 = layer_dict["block1_pool"].output
    pool2 = layer_dict["block2_pool"].output
    pool3 = layer_dict["block3_pool"].output
    pool4 = layer_dict["block4_pool"].output
    pool5 = layer_dict["block5_pool"].output

    def continue_from(tensor, start_layer_name: str):
        """Re-feed tensor through the base graph starting at start_layer_name."""
        take = False
        x = tensor
        for l in base.layers:
            if l.name == start_layer_name:
                take = True
            if take:
                x = l(x)
        return x

    se_name_map = {
        "before_conv1_1":        ("se_at_input",   inputs, "block1_conv1"),
        "between_pool1_conv2_1": ("se_after_pool1", pool1,  "block2_conv1"),
        "between_pool2_conv3_1": ("se_after_pool2", pool2,  "block3_conv1"),
        "between_pool3_conv4_1": ("se_after_pool3", pool3,  "block4_conv1"),
        "between_pool4_conv5_1": ("se_after_pool4", pool4,  "block5_conv1"),
    }

    if position in se_name_map:
        se_name, tap, resume = se_name_map[position]
        x = se_block(tap, ratio=se_ratio, name=se_name)
        x = continue_from(x, resume)
    else:  # between_pool5_dense
        x = se_block(pool5, ratio=se_ratio, name="se_after_pool5")

    outputs = build_classifier_head(x, name="cls")
    model = models.Model(inputs=inputs, outputs=outputs, name=f"vgg16_se_{position}")
    for l in model.layers:
        if l.name.startswith("block"):
            l.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
