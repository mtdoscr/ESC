# A Self-Supervised Pre-training Framework for Environmental Sound Classification

This project provides a PyTorch implementation of the Wav2Vec 2.0 framework for self-supervised learning on audio data. The model is pre-trained on the UrbanSound8K dataset and then fine-tuned for the task of urban sound classification.

## Features

- **Wav2Vec 2.0**: A complete implementation of the Wav2Vec 2.0 model, including the multi-layer CNN feature extractor, Transformer context network, and quantization mechanism.
- **Self-Supervised Pre-training**: A script to perform contrastive pre-training on unlabelled audio data (UrbanSound8K).
- **Fine-tuning for Classification**: A script to fine-tune the pre-trained model for sound classification.
- **Experiment Tracking**: Integrated with TensorBoard for easy monitoring of losses and metrics.
- **GPU Support**: Configured to run on Intel GPUs (`xpu`) and easily adaptable for NVIDIA GPUs (`cuda`).

## File Structure

The project is organized as follows:

```
.
├── Wav2vec2_model.py       # Defines the core Wav2Vec 2.0 model architecture.
├── dataset.py              # PyTorch Dataset class for pre-training.
├── pre-train.py            # Script to run the self-supervised pre-training.
├── finetune.py             # Script to fine-tune the model and run classification experiments.
│
├── UrbanSound8K/           # Directory for the dataset (must be downloaded).
│   ├── UrbanSound8K.csv
│   └── fold1/
│       └── ...
│
├── checkpoints/            # Stores saved model checkpoints from pre-training.
│   ├── pretrain_checkpoint_best.pth
│   └── pretrain_checkpoint_latest.pth
│
├── runs/                   # Contains TensorBoard logs for all training runs.
│   ├── finetuning/
│   └── pretrain_.../
│
├── assets/                 # Contains output artifacts like confusion matrices.
│   └── Confusion_matrix_.../
│
├── finetuning_comparison_...csv # CSV files summarizing the results of fine-tuning experiments.
└── Results.txt             # Raw log output from experiment runs.
```

## Setup and Installation

### 1. Dependencies

This project requires Python and PyTorch. You can install the necessary libraries using pip. It is recommended to use a virtual environment.

```bash
pip install torch torchaudio pandas numpy scikit-learn seaborn matplotlib tensorboard
```

### 2. Dataset (UrbanSound8K)

You need to download the UrbanSound8K dataset and place it in the root of the project directory.

- **Download Link**: You can download the dataset from the official website: [Urban Sound Datasets](https://urbansounddataset.weebly.com/urbansound8k.html).
- **Setup**: After downloading, extract the archive. You should have a folder named `UrbanSound8K` in your project root, containing the metadata file `UrbanSound8K.csv` and the audio folders (`fold1`, `fold2`, etc.).

### 3. GPU Configuration

The code is written to use an **Intel GPU (`xpu`)** by default. If you are using an **NVIDIA GPU**, you need to change the device setting.

- Open the following files: `pre-train.py` and `finetune.py`.
- Find the line `device = 'xpu' if torch.xpu.is_available() else 'cpu'`.
- Change it to `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.

## How to Run

The workflow is divided into two main stages: pre-training and fine-tuning.

### Step 1: Self-Supervised Pre-training

First, run the pre-training script to learn audio representations from the UrbanSound8K dataset. This process does not use any labels.

```bash
python pre-train.py
```

This script will:

- Train the `Wav2Vec2PretrainModel` for the number of epochs specified in the script.
- Save TensorBoard logs in the `runs/` directory.
- Save the best-performing model checkpoint to `checkpoints/pretrain_checkpoint_best.pth` based on validation loss.

### Step 2: Fine-tuning for Classification

After pre-training, you can fine-tune the model for sound classification. The `finetune.py` script provides several modes for this.

The script compares four different models:

1. **Finetune_Full_Wav2Vec2**: Fine-tunes the entire pre-trained model.
2. **Finetune_FeatureExtractor**: Uses only the pre-trained CNN feature extractor.
3. **Finetune_FeatureExtractor_Quantizer**: Uses the CNN feature extractor and the quantizer.
4. **Baseline_From_Scratch**: Trains a model with the same architecture as `Finetune_FeatureExtractor` , but no pre-trained weights.

You can run experiments using different command-line arguments:

**Example 1: Fine-tuning on the Full Dataset**

This mode fine-tunes directly on all 10 classes of the UrbanSound8K dataset.

```bash
python finetune.py --mode full --epochs 30
```

**Example 2: Fine-tuning on a Subset**

This mode runs a single experiment on a specified subset of classes.

```bash
python finetune.py --mode subset --subset_classes 0 1 2 --epochs 25
```

**Example 3. Finetune on a subset and on the full dataset, and compare the results:**

```bash
python finetune.py --mode subset_then_full --subset_classes 3 4 9 --epochs 20
```

- `--subset_classes 3 4 9`: Defines which classes to use for the first phase (3: dog_bark, 4: drilling, 9: street_music).
- `--epochs 20`: Sets the number of epochs for *each* phase.

### Viewing Results

All training runs (both pre-training and fine-tuning) are logged to the `runs/` directory. You can visualize the training progress, including losses and validation metrics, using TensorBoard.

To launch TensorBoard, run the following command in your terminal:

```bash
tensorboard --logdir runs
```

Navigate to `http://localhost:6006/` in your web browser to view the dashboard.

### Output Files

- **Final Metrics**: The results of the fine-tuning experiments are saved in `.csv` files (e.g., `finetuning_comparison_subset_then_full.csv`). These tables provide a side-by-side comparison of the different models.
- **Confusion Matrices**: Visualizations of the model performance on the test set are saved as PNG images in the `assets/` directory.
- **Raw Logs**: The complete console output of the training scripts is captured in `Results.txt` for detailed inspection.
