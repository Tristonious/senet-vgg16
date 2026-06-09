# SENet Attention Modules in VGG16

Squeeze-and-Excitation (SE) attention blocks integrated into a pretrained VGG16 at six insertion points, trained and evaluated on Imagenette. Course project for CSCI 8110 comparing baseline and SE-augmented classification accuracy across insertion positions, with feature-map visualization before and after recalibration.

---

## Architecture

| Component | Design |
|---|---|
| Backbone | VGG16 pretrained on ImageNet (include_top=False); conv weights frozen |
| SE block | Squeeze (GAP) → Excitation (Dense-ReLU-Dense-Sigmoid) → Scale (channel-wise multiply) |
| SE ratio | 16 (bottleneck reduction factor) |
| Classifier head | GAP → Dense(256, ReLU) → Dropout(0.3) → Dense(10, Softmax) |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical cross-entropy |
| Dataset | Imagenette (10-class subset of ImageNet) |
| Input size | 160×160 (reduced from 224×224 for training speed) |

### SE Insertion Positions

| Position key | Location in VGG16 |
|---|---|
| `before_conv1_1` | Between input and block1_conv1 |
| `between_pool1_conv2_1` | After block1_pool, before block2_conv1 |
| `between_pool2_conv3_1` | After block2_pool, before block3_conv1 |
| `between_pool3_conv4_1` | After block3_pool, before block4_conv1 |
| `between_pool4_conv5_1` | After block4_pool, before block5_conv1 |
| `between_pool5_dense` | After block5_pool, before classifier head |

---

## Results

| Model | Val Accuracy |
|---|---|
| Baseline (no SE) | 0.955 |
| SE — before_conv1_1 | 0.821 |
| SE — between_pool1_conv2_1 | 0.893 |
| SE — between_pool2_conv3_1 | 0.746 |
| SE — between_pool3_conv4_1 | 0.844 |
| SE — between_pool4_conv5_1 | 0.924 |
| SE — between_pool5_dense | 0.906 |

SE did not consistently improve over the frozen-backbone baseline. Early insertions (before pool2) degraded accuracy most severely, while later insertions (pool4–pool5) retained accuracy above 0.90. This suggests channel recalibration is more effective at high-level semantic layers than at low-level texture extraction stages.

---

## Figures

| Figure | Description |
|---|---|
| ![Comparison bar chart](figures/before_after_grouped_bar.png) | Baseline vs SE validation accuracy at each insertion point |

Feature map visualizations (baseline and SE after pool3) are generated to `figures/` at runtime.

---

## Project Structure

```
senet-vgg16/
├── run.py                      # Entry point: trains all models, saves figures
├── src/
│   ├── __init__.py
│   ├── model.py                # SE block, VGG16 backbone, model assembly
│   ├── data.py                 # Imagenette loading and preprocessing
│   ├── train.py                # Training and evaluation loop
│   ├── viz.py                  # Feature map visualization, comparison bar chart
│   └── metrics.py              # Feature-map quality metrics (entropy, focus)
├── figures/                    # Committed: comparison bar chart; runtime figures added here
├── results/                    # Gitignored: JSON history/metrics, model checkpoints
├── models/                     # Gitignored: saved .keras checkpoints
├── docs/
│   └── paper.pdf
├── requirements.txt
└── .gitignore
```

---

## Usage

### 1. Download Imagenette

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xf imagenette2-320.tgz
```

Then either place the extracted `imagenette2-320/` folder next to the repo, or set an environment variable:

```bash
export IMAGENETTE_PATH=/path/to/imagenette2-320
```

### 2. Install dependencies and run

```bash
pip install -r requirements.txt
python run.py
```

Training defaults to 500 train / 250 val samples for quick iteration. To use the full dataset, set `MAX_TRAIN = None` and `MAX_VAL = None` in `run.py`.

---

## AI Assistance Note

The original implementation for this project was developed as coursework for CSCI 8110 at the University of Nebraska Omaha. The code in this repository has been refactored with the assistance of Claude (Anthropic) for clarity, modularity, and readability. The SE block design, insertion strategy, training configuration, experimental comparisons, and analysis are my own work.

---

## Paper

Covers SE block implementation, six-position insertion experiment, accuracy comparison across positions, feature-map visualization analysis, and discussion of attention placement effects on frozen VGG16.

[`Project Paper`](docs/paper.pdf)

---

## References

1. CSCI 8110 Lecture Notes, "Attention Modules (Lectures 13 and 14)," University of Nebraska Omaha, 2025.
2. Keras Applications — VGG16. https://keras.io/api/applications/vgg/#vgg16-function
3. TensorFlow API Documentation. https://www.tensorflow.org/api_docs
4. fastai, "Imagenette Dataset." https://github.com/fastai/imagenette
