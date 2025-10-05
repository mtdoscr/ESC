import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
import warnings
warnings.filterwarnings("ignore") # Suppress warnings
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
import argparse

# ==============================================================================
# This script is for fine-tuning a pre-trained Wav2Vec2 model on the UrbanSound8K dataset for sound classification.
# It defines several model variations for comparison, a data loader, and a training/evaluation pipeline.
#
# The Wav2Vec2 model architecture is re-defined here for self-containment, allowing this script to run independently
# of the pre-training script, provided the model state dictionary is available.
# ==============================================================================


# --- Pre-trained Model Architecture Definition ---
# The following classes define the building blocks of the Wav2Vec2 model,
# mirroring the architecture from `Wav2vec2_model.py`.

class ConvLayerBlock(nn.Module):
    """A convolutional block with optional LayerNorm, used in the FeatureExtractor."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, layer_norm=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=(kernel_size - 1) // 2)
        self.norm = nn.LayerNorm(out_channels) if layer_norm else nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self._layer_norm = layer_norm
    def forward(self, x):
        x = self.conv(x)
        if self._layer_norm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)
        return self.activation(x)

class SelfAttention(nn.Module):
    """Standard multi-head self-attention mechanism for the Transformer."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = dropout
    def forward(self, x, padding_mask=None):
        seq_len, batch_size, embed_dim = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_weights, v).transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim)
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network for the Transformer."""
    def __init__(self, io_features, interm_features, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(io_features, interm_features)
        self.linear2 = nn.Linear(interm_features, io_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer encoder."""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x, padding_mask=None):
        residual = x
        x = self.attention(x, padding_mask=padding_mask)
        x = self.norm1(residual + self.dropout1(x))
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout2(x))
        return x

class FeatureExtractor(nn.Module):
    """Extracts latent feature representations from raw audio waveforms using CNNs."""
    def __init__(self, conv_layer_configs: List[Tuple[int, int, int]]):
        super().__init__()
        in_channels, layers = 1, []
        for i, (out_channels, k, s) in enumerate(conv_layer_configs):
            layers.append(ConvLayerBlock(in_channels, out_channels, k, s, layer_norm=i > 0))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)

class TransformerEncoder(nn.Module):
    """The main Transformer encoder stack."""
    def __init__(self, in_features, embed_dim, proj_dropout, pos_conv_kernel, pos_conv_groups, num_layers, num_heads, ffn_dim, dropout, attention_dropout, layer_drop):
        super().__init__()
        self.layer_drop = layer_drop
        self.proj_norm = nn.LayerNorm(in_features)
        self.proj_linear = nn.Linear(in_features, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.pos_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=pos_conv_kernel, padding=pos_conv_kernel // 2, groups=pos_conv_groups)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_activation = nn.GELU()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout, attention_dropout) for _ in range(num_layers)])
    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        padding_mask = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            padding_mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        x = self.proj_dropout(self.proj_linear(self.proj_norm(x)))
        
        pos_x = x.transpose(1, 2)
        pos_emb = self.pos_conv(pos_x)
        pos_emb = self.pos_activation(pos_emb)
        pos_emb = pos_emb[..., :x.shape[1]]
        pos_emb = pos_emb.transpose(1, 2)

        x = x + pos_emb
        x = x.transpose(0, 1)
        for layer in self.layers:
            if not self.training or torch.rand(1).item() > self.layer_drop:
                x = layer(x, padding_mask=padding_mask)
        return x.transpose(0, 1)

class Wav2Vec2Model(nn.Module):
    """The core Wav2Vec2 model, combining the feature extractor and Transformer encoder."""
    def __init__(self, feature_extractor, encoder):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
    def _get_output_lengths(self, input_lengths):
        def _conv_out_len(in_len, k, s, p):
            return torch.floor((in_len + 2 * p - k) / s) + 1
        lengths = input_lengths.float()
        for layer in self.feature_extractor.layers:
            k = layer.conv.kernel_size[0]
            s = layer.conv.stride[0]
            p = layer.conv.padding[0]
            lengths = _conv_out_len(lengths, k, s, p)
        return lengths.to(torch.long)
    def forward(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        features = self.feature_extractor(waveforms)
        output_lengths = self._get_output_lengths(lengths) if lengths is not None else None
        encoded_features = self.encoder(features, lengths=output_lengths)
        return encoded_features, output_lengths

class Quantizer(nn.Module):
    """The codebook quantizer, used in one of the fine-tuning strategies."""
    def __init__(self, input_dim: int, num_groups: int, num_vars: int):
        super().__init__()
        assert input_dim % num_groups == 0, "input_dim must be divisible by num_groups"
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.codebooks = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, input_dim // num_groups))
        nn.init.uniform_(self.codebooks)
        self.logit_proj = nn.Linear(input_dim, num_groups * num_vars)
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, feat_dim = x.shape
        logits = self.logit_proj(x).view(batch_size * seq_len * self.num_groups, self.num_vars)
        quantized_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        codevectors = self.codebooks.view(self.num_groups, self.num_vars, -1)
        quantized_one_hot = quantized_one_hot.view(batch_size * seq_len, self.num_groups, self.num_vars)
        quantized_vectors = (quantized_one_hot.unsqueeze(-2) @ codevectors).squeeze(-2)
        quantized_vectors = quantized_vectors.view(batch_size, seq_len, -1)
        probs = torch.softmax(logits.view(batch_size * seq_len, self.num_groups, self.num_vars), dim=-1)
        avg_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)).mean()
        return quantized_vectors, perplexity

def wav2vec2_base_model():
    """Helper function to build a Wav2Vec2 model with the 'base' configuration."""
    conv_layer_configs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    feature_extractor = FeatureExtractor(conv_layer_configs)
    encoder = TransformerEncoder(
        in_features=512, embed_dim=768, proj_dropout=0.1, pos_conv_kernel=128,
        pos_conv_groups=16, num_layers=12, num_heads=12, ffn_dim=3072,
        dropout=0.1, attention_dropout=0.1, layer_drop=0.1,
    )
    model = Wav2Vec2Model(feature_extractor, encoder)
    return model

# --- Model 1: Fine-tune Full Pre-trained Model ---
class Wav2Vec2ForClassification(nn.Module):
    """
    Fine-tunes the entire pre-trained Wav2Vec2 model.
    The feature extractor is partially frozen (only last 2 layers are trained),
    while the Transformer encoder is fully trained. A new classification head is added.
    """
    def __init__(self, pretrained_model_path, num_classes):
        super().__init__()
        self.wav2vec2 = wav2vec2_base_model()
        try:
            # Load weights from the pre-trained model checkpoint
            pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))['model_state_dict']
            # Filter for wav2vec_model weights and remove the prefix
            wav2vec_dict = {k.replace('wav2vec_model.', ''): v for k, v in pretrained_dict.items() if k.startswith('wav2vec_model.')}
            self.wav2vec2.load_state_dict(wav2vec_dict)
        except FileNotFoundError:
            print(f"Warning: Pre-trained model path {pretrained_model_path} not found. Initializing from scratch.")
        
        # Freeze most of the feature extractor
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        # Unfreeze the last two convolutional layers for fine-tuning
        for param in self.wav2vec2.feature_extractor.layers[-2:].parameters():
            param.requires_grad = True
        # The entire Transformer encoder is trainable
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = True
        
        # Add a new linear layer for classification
        self.classification_head = nn.Linear(768, num_classes) # 768 is the Transformer's output dim

    def forward(self, x):
        # Get contextualized features from the Wav2Vec2 model
        context_features, _ = self.wav2vec2(x)
        # Pool the features over the time dimension (mean pooling)
        pooled_features = torch.mean(context_features, dim=1)
        # Classify the pooled features
        return self.classification_head(pooled_features)

# --- Model 2: Fine-tune on Feature Extractor ---
class Wav2Vec2FeatureExtractorForClassification(nn.Module):
    """
    Uses only the pre-trained Feature Extractor part of Wav2Vec2.
    The feature extractor is frozen except for the last two layers.
    A new, simple classification head is trained on top of these features.
    """
    def __init__(self, pretrained_model_path, num_classes):
        super().__init__()
        full_model = wav2vec2_base_model()
        try:
            # Load weights from the pre-trained model checkpoint
            pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))['model_state_dict']
            wav2vec_dict = {k.replace('wav2vec_model.', ''): v for k, v in pretrained_dict.items() if k.startswith('wav2vec_model.')}
            full_model.load_state_dict(wav2vec_dict)
        except FileNotFoundError:
            print(f"Warning: Pre-trained model path {pretrained_model_path} not found. Initializing from scratch.")

        self.feature_encoder = full_model.feature_extractor
        # Freeze most of the feature extractor
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        # Unfreeze the last two convolutional layers for fine-tuning
        for param in self.feature_encoder.layers[-2:].parameters():
            param.requires_grad = True
            
        # A simple classification head with pooling
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_classes) # 512 is the feature extractor's output dim
        )
    def forward(self, x):
        features = self.feature_encoder(x)
        # Transpose for pooling: (Batch, Time, Features) -> (Batch, Features, Time)
        features = features.transpose(1, 2)
        return self.classification_head(features)

# --- Model 3: Fine-tune on Feature Extractor + Quantizer ---
class Wav2Vec2FeatureExtractorQuantizerForClassification(nn.Module):
    """
    Uses the pre-trained Feature Extractor and Quantizer.
    The feature extractor is partially frozen, and the quantizer is fully frozen.
    A new classification head is trained on the quantized features.
    """
    def __init__(self, pretrained_model_path, num_classes):
        super().__init__()
        full_model = wav2vec2_base_model()
        self.quantizer = Quantizer(input_dim=512, num_groups=2, num_vars=320)
        try:
            # Load weights for both the main model and the quantizer
            pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))['model_state_dict']
            wav2vec_dict = {k.replace('wav2vec_model.', ''): v for k, v in pretrained_dict.items() if k.startswith('wav2vec_model.')}
            full_model.load_state_dict(wav2vec_dict)
            quantizer_dict = {k.replace('quantizer.', ''): v for k, v in pretrained_dict.items() if k.startswith('quantizer.')}
            self.quantizer.load_state_dict(quantizer_dict)
        except FileNotFoundError:
            print(f"Warning: Pre-trained model path {pretrained_model_path} not found. Initializing from scratch.")

        self.feature_encoder = full_model.feature_extractor
        # Freeze most of the feature extractor
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        # Unfreeze the last two convolutional layers
        for param in self.feature_encoder.layers[-2:].parameters():
            param.requires_grad = True
        # Freeze the quantizer completely
        for param in self.quantizer.parameters():
            param.requires_grad = False
        
        # Add a new linear layer for classification
        self.classification_head = nn.Linear(512, num_classes) # 512 is the quantizer's output dim

    def forward(self, x):
        features = self.feature_encoder(x)
        # Get quantized representations
        quantized, _ = self.quantizer(features)
        # Pool the quantized features over the time dimension
        pooled_features = torch.mean(quantized, dim=1)
        return self.classification_head(pooled_features)

# --- Model 4: Baseline (Train from Scratch) ---
class BaselineFeatureExtractorForClassification(nn.Module):
    """
    A baseline model that uses the same Feature Extractor architecture as Wav2Vec2
    but is trained from scratch, without any pre-trained weights.
    """
    def __init__(self, num_classes):
        super().__init__()
        # Define the same feature extractor architecture
        conv_layer_configs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        self.feature_encoder = FeatureExtractor(conv_layer_configs)
        # All parameters are trainable by default
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.feature_encoder(x)
        features = features.transpose(1, 2)
        return self.classification_head(features)

# --- Dataset Definition for Fine-tuning ---
class UrbanSoundDataset(Dataset):
    """
    A PyTorch Dataset for UrbanSound8K, adapted for classification.
    It loads audio, processes it, and returns the waveform and its class label.
    """
    def __init__(self, annotations_df, audio_dir, target_sr=16000, target_len_samples=64000, class_mapping=None):
        self.annotations = annotations_df
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.target_len_samples = target_len_samples
        self.class_mapping = class_mapping
        # If a class mapping is provided (for subset training), apply it
        if self.class_mapping:
            self.annotations['classID'] = self.annotations['classID'].map(self.class_mapping)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Construct file path from metadata
        audio_path = os.path.join(self.audio_dir, 'fold' + str(self.annotations.iloc[idx, 5]), self.annotations.iloc[idx, 0])
        # Load and process audio
        waveform, sr = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
        waveform = resampler(waveform)
        if waveform.shape[0] > 1: # Convert to mono
            waveform = torch.mean(waveform, dim=0)
        waveform = waveform.squeeze(0)
        # Pad or truncate to a fixed length
        if waveform.shape[0] > self.target_len_samples:
            waveform = waveform[:self.target_len_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_len_samples - waveform.shape[0]))
        # Return waveform and its corresponding class ID
        return waveform, self.annotations.iloc[idx, 6]

# --- Training and Evaluation Function ---
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, device, writer, epochs, learning_rate, class_names, phase=""):
    """
    Handles the complete training and evaluation pipeline for a given model.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Use AdamW optimizer, only on parameters that are not frozen
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(f"--- Training {model_name} ({phase}) ---")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar(f'Loss/train_{phase}', epoch_loss, epoch)

        # --- Validation Phase ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        writer.add_scalar(f'Accuracy/validation_{phase}', val_accuracy, epoch)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        scheduler.step()

    print(f'Finished Training {model_name} ({phase})')

    # --- Test Phase ---
    print(f'--- Evaluating on Test Set for {model_name} ({phase}) ---')
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Calculate and print metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    # Binarize labels for multi-class AUC calculation
    all_labels_bin = label_binarize(all_labels, classes=range(len(class_names)))
    try:
        auc = roc_auc_score(all_labels_bin, np.array(all_probs), multi_class='ovr')
    except ValueError as e:
        print(f"Could not calculate AUC: {e}")
        auc = 0.0

    print(f'--- Results for {model_name} ({phase}) ---')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test F1 Score (Macro): {f1:.4f}')
    print(f'Test AUC (OvR): {auc:.4f}')

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix - {model_name} ({phase})')
    cm_path = f'confusion_matrix_{model_name}_{phase}.png'
    plt.savefig(cm_path); plt.close()
    print(f'Confusion matrix saved to {cm_path}')

    # Return a dictionary of the final metrics
    return {'Model': model_name, 'Phase': phase, 'Accuracy': accuracy, 'F1_Score': f1, 'AUC_OvR': auc}

def get_data_loaders(annotations_df, audio_dir, batch_size, target_classes=None):
    """
    Creates and returns train, validation, and test DataLoaders.
    Splits data differently based on whether a subset of classes is used.
    """
    if target_classes:
        # --- Subset Mode ---
        # Map selected class IDs to a new range (0, 1, 2, ...)
        class_mapping = {original_class: new_class for new_class, original_class in enumerate(target_classes)}
        annotations = annotations_df[annotations_df['classID'].isin(target_classes)].reset_index(drop=True)
        
        # Stratified split into train (70%), validation (15%), and test (15%)
        train_df, temp_df = train_test_split(annotations, test_size=0.3, random_state=42, stratify=annotations['classID'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['classID'])

        train_dataset = UrbanSoundDataset(train_df, audio_dir, class_mapping=class_mapping)
        val_dataset = UrbanSoundDataset(val_df, audio_dir, class_mapping=class_mapping)
        test_dataset = UrbanSoundDataset(test_df, audio_dir, class_mapping=class_mapping)
    else:
        # --- Full Dataset Mode ---
        # Use the standard fold-based split for UrbanSound8K
        train_df = annotations_df[annotations_df['fold'].isin(range(1, 9))].reset_index(drop=True)
        val_df = annotations_df[annotations_df['fold'] == 9].reset_index(drop=True)
        test_df = annotations_df[annotations_df['fold'] == 10].reset_index(drop=True)
        class_mapping = None

        train_dataset = UrbanSoundDataset(train_df, audio_dir)
        val_dataset = UrbanSoundDataset(val_df, audio_dir)
        test_dataset = UrbanSoundDataset(test_df, audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Fine-tune Wav2Vec2 on UrbanSound8K.')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'subset', 'subset_then_full'],
                        help='Finetuning mode: full dataset, subset, or subset then full.')
    parser.add_argument('--subset_classes', type=int, nargs='+', default=[3, 4, 9],
                        help='List of class IDs to use for subset mode (e.g., dog_bark, drilling, street_music).')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer.')
    
    args = parser.parse_args()

    # --- Configuration ---
    AUDIO_DIR = 'UrbanSound8K'
    ANNOTATIONS_FILE = os.path.join(AUDIO_DIR, 'UrbanSound8K.csv')
    PRETRAINED_MODEL_PATH = 'checkpoints/pretrain_checkpoint_best.pth'
    
    ALL_CLASS_NAMES = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 
        'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
    ]
    
    device = 'xpu' if torch.xpu.is_available() else 'cpu'
    print(f"Using device: {device}")

    annotations = pd.read_csv(ANNOTATIONS_FILE)
    results = [] # To store metrics from all runs

    # --- Experiment Execution ---
    if args.mode == 'full' or args.mode == 'subset':
        # Determine classes and names based on mode
        target_classes = list(range(10)) if args.mode == 'full' else args.subset_classes
        class_names = ALL_CLASS_NAMES if args.mode == 'full' else [ALL_CLASS_NAMES[i] for i in args.subset_classes]
        num_classes = len(target_classes)

        # Get data loaders for the specified mode
        train_loader, val_loader, test_loader = get_data_loaders(annotations, AUDIO_DIR, args.batch_size, target_classes if args.mode == 'subset' else None)

        # Define the dictionary of models to be trained and evaluated
        models_to_train = {
            "1_Finetune_Full_Wav2Vec2": Wav2Vec2ForClassification(PRETRAINED_MODEL_PATH, num_classes=num_classes),
            "2_Finetune_FeatureExtractor": Wav2Vec2FeatureExtractorForClassification(PRETRAINED_MODEL_PATH, num_classes=num_classes),
            "3_Finetune_FeatureExtractor_Quantizer": Wav2Vec2FeatureExtractorQuantizerForClassification(PRETRAINED_MODEL_PATH, num_classes=num_classes),
            "4_Baseline_From_Scratch": BaselineFeatureExtractorForClassification(num_classes=num_classes)
        }

        # Iterate through each model, train, evaluate, and store results
        for model_name, model in models_to_train.items():
            log_dir = os.path.join('runs', 'finetuning', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}_{args.mode}")
            writer = SummaryWriter(log_dir)
            print(f"Tensorboard log directory: {log_dir}")
            
            metrics = train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, device, writer, args.epochs, args.learning_rate, class_names, phase=args.mode)
            results.append(metrics)
            writer.close()

    elif args.mode == 'subset_then_full':
        # --- Phase 1: Train on a subset of classes ---
        print("\n--- PHASE 1: Training on Subset ---")
        subset_class_names = [ALL_CLASS_NAMES[i] for i in args.subset_classes]
        num_subset_classes = len(args.subset_classes)
        
        subset_train_loader, subset_val_loader, subset_test_loader = get_data_loaders(annotations, AUDIO_DIR, args.batch_size, target_classes=args.subset_classes)

        models_to_train = {
            "1_Finetune_Full_Wav2Vec2": Wav2Vec2ForClassification(PRETRAINED_MODEL_PATH, num_classes=num_subset_classes),
            "2_Finetune_FeatureExtractor": Wav2Vec2FeatureExtractorForClassification(PRETRAINED_MODEL_PATH, num_classes=num_subset_classes),
            "3_Finetune_FeatureExtractor_Quantizer": Wav2Vec2FeatureExtractorQuantizerForClassification(PRETRAINED_MODEL_PATH, num_classes=num_subset_classes),
            "4_Baseline_From_Scratch": BaselineFeatureExtractorForClassification(num_classes=num_subset_classes)
        }

        trained_models = {}
        for model_name, model in models_to_train.items():
            log_dir = os.path.join('runs', 'finetuning', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}_subset_phase")
            writer = SummaryWriter(log_dir)
            print(f"Tensorboard log directory: {log_dir}")
            
            metrics = train_and_evaluate(model, model_name, subset_train_loader, subset_val_loader, subset_test_loader, device, writer, args.epochs, args.learning_rate, subset_class_names, phase="subset")
            results.append(metrics)
            trained_models[model_name] = model # Save the trained model for the next phase
            writer.close()

        # --- Phase 2: Continue training on the full dataset ---
        print("\n--- PHASE 2: Transfer Training to Full Dataset ---")
        full_class_names = ALL_CLASS_NAMES
        num_full_classes = len(full_class_names)
        full_train_loader, full_val_loader, full_test_loader = get_data_loaders(annotations, AUDIO_DIR, args.batch_size, target_classes=None)

        for model_name, model in trained_models.items():
            # Replace the classification head with a new one for the full number of classes
            if hasattr(model, 'classification_head'):
                if isinstance(model.classification_head, nn.Linear):
                    in_features = model.classification_head.in_features
                    model.classification_head = nn.Linear(in_features, num_full_classes)
                elif isinstance(model.classification_head, nn.Sequential):
                    # Find and replace the final linear layer in a sequential head
                    final_layer = model.classification_head[-1]
                    if isinstance(final_layer, nn.Linear):
                        in_features = final_layer.in_features
                        model.classification_head[-1] = nn.Linear(in_features, num_full_classes)

            log_dir = os.path.join('runs', 'finetuning', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}_full_phase")
            writer = SummaryWriter(log_dir)
            print(f"Tensorboard log directory: {log_dir}")

            # Continue training and evaluate on the full dataset
            metrics = train_and_evaluate(model, model_name, full_train_loader, full_val_loader, full_test_loader, device, writer, args.epochs, args.learning_rate, full_class_names, phase="full")
            results.append(metrics)
            writer.close()

    # --- Save Final Results ---
    results_df = pd.DataFrame(results)
    results_filename = f'finetuning_comparison_{args.mode}.csv'
    results_df.to_csv(results_filename, index=False)
    print("\n--- Final Comparison Results ---")
    print(results_df)
    print(f"\nResults table saved to {results_filename}")