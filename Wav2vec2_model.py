import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# ==============================================================================
# This script defines the architecture of a Wav2Vec2 model from scratch in PyTorch.
# It includes the feature extractor (CNNs), the context network (Transformer),
# and the components required for the self-supervised pre-training objective,
# such as the quantizer and contrastive loss.
#
# Based on the paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
# (https://arxiv.org/abs/2006.11477)
# ==============================================================================


# ==============================================================================
# 1. Core Model Architecture
# ==============================================================================

class ConvLayerBlock(nn.Module):
    """
    A single convolutional block for the feature extractor.
    It consists of a 1D convolution, a normalization layer (GroupNorm or LayerNorm), and a GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, layer_norm=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            bias=bias, 
            padding=(kernel_size - 1) // 2 # 'SAME' padding
        )
        self.norm = nn.LayerNorm(out_channels) if layer_norm else nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self._layer_norm = layer_norm

    def forward(self, x):
        x = self.conv(x)
        # Transpose for LayerNorm to operate on the feature dimension
        if self._layer_norm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)
        return self.activation(x)

class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module as used in Transformers.
    It computes scaled dot-product attention over multiple heads.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for query, key, and value
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = dropout

    def forward(self, x, padding_mask=None):
        # Input shape: (SeqLen, Batch, Dim)
        seq_len, batch_size, embed_dim = x.shape
        
        # 1. Project to Q, K, V and split into heads
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        # 2. Calculate attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        
        # 3. Apply padding mask
        if padding_mask is not None:
            # Reshape to (Batch, Heads, SeqLen, SeqLen) to apply mask
            attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 
                float("-inf")
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, seq_len)
            
        # 4. Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 5. Apply attention to values
        attn_output = torch.bmm(attn_weights, v)
        
        # 6. Concatenate heads and project out
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim)
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """
    The position-wise Feed-Forward Network (FFN) used in each Transformer layer.
    """
    def __init__(self, io_features, interm_features, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(io_features, interm_features)
        self.linear2 = nn.Linear(interm_features, io_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder, combining self-attention and a feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        # Self-Attention block
        residual = x
        x = self.attention(x, padding_mask=padding_mask)
        x = self.norm1(residual + self.dropout1(x))
        
        # Feed-Forward block
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout2(x))
        return x

class FeatureExtractor(nn.Module):
    """
    The CNN-based feature extractor that processes the raw audio waveform.
    It downsamples the audio and creates a sequence of feature vectors.
    """
    def __init__(self, conv_layer_configs: List[Tuple[int, int, int]]):
        super().__init__()
        in_channels, layers = 1, []
        for i, (out_channels, kernel_size, stride) in enumerate(conv_layer_configs):
            # The first layer uses GroupNorm, subsequent layers use LayerNorm
            layers.append(ConvLayerBlock(in_channels, out_channels, kernel_size, stride, layer_norm=(i > 0)))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Input shape: (Batch, SeqLen)
        x = x.unsqueeze(1) # (Batch, 1, SeqLen)
        for layer in self.layers:
            x = layer(x)
        # Output shape: (Batch, Features, SeqLen_reduced) -> (Batch, SeqLen_reduced, Features)
        return x.transpose(1, 2)

class TransformerEncoder(nn.Module):
    """
    The context network, composed of a stack of Transformer encoder layers.
    It takes the features from the FeatureExtractor and produces contextualized representations.
    """
    def __init__(self, in_features, embed_dim, proj_dropout, pos_conv_kernel, pos_conv_groups, num_layers, num_heads, ffn_dim, dropout, attention_dropout, layer_drop):
        super().__init__()
        self.layer_drop = layer_drop
        
        # Project features to the Transformer's embedding dimension
        self.proj_norm = nn.LayerNorm(in_features)
        self.proj_linear = nn.Linear(in_features, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        # Positional embedding using a 1D convolution
        self.pos_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=pos_conv_kernel, padding=pos_conv_kernel // 2, groups=pos_conv_groups)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_activation = nn.GELU()
        
        # Stack of Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout, attention_dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        # Create a padding mask from the original lengths
        padding_mask = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            padding_mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
            
        # Project features
        x = self.proj_dropout(self.proj_linear(self.proj_norm(x)))
        
        # Add positional embeddings
        pos_x = x.transpose(1, 2)
        pos_emb = self.pos_conv(pos_x)
        pos_emb = self.pos_activation(pos_emb)
        pos_emb = pos_emb[..., :x.shape[1]] # Crop to match input sequence length
        pos_emb = pos_emb.transpose(1, 2)
        x = x + pos_emb
        
        # The Transformer expects (SeqLen, Batch, Dim)
        x = x.transpose(0, 1)
        
        # Pass through Transformer layers with LayerDrop
        for layer in self.layers:
            if not self.training or torch.rand(1).item() > self.layer_drop:
                x = layer(x, padding_mask=padding_mask)
                
        # Return to (Batch, SeqLen, Dim)
        return x.transpose(0, 1)

class Wav2Vec2Model(nn.Module):
    """
    The main Wav2Vec2 model, combining the feature extractor and the Transformer encoder.
    """
    def __init__(self, feature_extractor, encoder):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculates the sequence length after passing through the convolutional feature extractor."""
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
        # 1. Extract features from raw audio
        features = self.feature_extractor(waveforms)
        
        # 2. Calculate the new lengths of the sequences after convolution
        output_lengths = self._get_output_lengths(lengths) if lengths is not None else None
        
        # 3. Get contextualized representations from the Transformer
        encoded_features = self.encoder(features, lengths=output_lengths)
        
        return encoded_features, output_lengths


# ==============================================================================
# 2. Pre-training Objective Components
# ==============================================================================

class Quantizer(nn.Module):
    """
    Implements product quantization using Gumbel-Softmax.
    This module learns a discrete set of representations (codebook vectors) for the latent features.
    These discrete representations serve as the targets in the contrastive pre-training task.
    """
    def __init__(self, input_dim: int, num_groups: int, num_vars: int):
        super().__init__()
        assert input_dim % num_groups == 0, "input_dim must be divisible by num_groups"
        self.input_dim = input_dim
        self.num_groups = num_groups # Number of codebooks (G)
        self.num_vars = num_vars     # Number of entries per codebook (V)
        
        # Codebooks: G groups, each with V entries of dimension D/G
        self.codebooks = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, input_dim // num_groups))
        nn.init.uniform_(self.codebooks)

        # Projection from latent features to logits for codebook selection
        self.logit_proj = nn.Linear(input_dim, num_groups * num_vars)

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        batch_size, seq_len, feat_dim = x.shape
        
        # Project features to logits and reshape for Gumbel-Softmax
        logits = self.logit_proj(x).view(batch_size * seq_len * self.num_groups, self.num_vars)
        
        # Differentiable discrete selection using Gumbel-Softmax
        # `hard=True` makes it return one-hot vectors, while being differentiable in the backward pass.
        quantized_one_hot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        
        # Retrieve the corresponding codebook vectors
        codevectors = self.codebooks.view(self.num_groups, self.num_vars, -1)
        quantized_one_hot = quantized_one_hot.view(batch_size * seq_len, self.num_groups, self.num_vars)
        
        # Use matrix multiplication to select codebook entries
        quantized_vectors = (quantized_one_hot.unsqueeze(-2) @ codevectors).squeeze(-2)
        quantized_vectors = quantized_vectors.view(batch_size, seq_len, -1) # Reshape back to (B, T, D)
        
        # Calculate codebook perplexity for the diversity loss
        # Perplexity is a measure of how uniformly the codebook entries are being used.
        probs = torch.softmax(logits.view(batch_size * seq_len, self.num_groups, self.num_vars), dim=-1)
        avg_probs = probs.mean(dim=0)  # Average over batch and time -> (G, V)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)).mean()
        
        return quantized_vectors, perplexity

class ContrastiveLoss(nn.Module):
    """
    Calculates the contrastive loss.
    The model must identify the correct quantized target vector (positive)
    from a set of distractors (negatives).
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, context_vectors: torch.Tensor, positive_targets: torch.Tensor, negative_targets: torch.Tensor):
        # context_vectors: (N, D), where N is total masked steps in batch
        # positive_targets: (N, D)
        # negative_targets: (N, K, D), where K is num_negatives
        
        # Combine positive and negative targets for similarity calculation
        targets = torch.cat([positive_targets.unsqueeze(1), negative_targets], dim=1) # Shape: (N, K+1, D)
        
        # Calculate cosine similarity between the context vector and all targets
        logits = F.cosine_similarity(context_vectors.unsqueeze(1), targets, dim=-1)
        
        # The target label is always index 0 (the positive sample)
        labels = torch.zeros(context_vectors.size(0), dtype=torch.long, device=context_vectors.device)
        
        # Calculate cross-entropy loss with temperature scaling
        return F.cross_entropy(logits / self.temperature, labels)

class DiversityLoss(nn.Module):
    """
    Calculates the diversity loss, which encourages the model to use all
    codebook entries equally. It is the negative of the codebook perplexity.
    """
    def forward(self, perplexity: torch.Tensor):
        # The paper's diversity loss is designed to maximize perplexity,
        # which is equivalent to minimizing its negative value.
        return -perplexity

class Wav2Vec2PretrainModel(nn.Module):
    """
    A wrapper model for the entire pre-training task.
    It combines the base Wav2Vec2 model, the quantizer, and the masking logic
    to produce the outputs needed for the contrastive and diversity losses.
    """
    def __init__(
        self,
        model: Wav2Vec2Model,
        quantizer: Quantizer,
        mask_prob: float = 0.065, # Probability of a timestep being the start of a mask
        mask_length: int = 10,    # Length of each mask span
        num_negatives: int = 100, # Number of negative samples for contrastive loss
    ):
        super().__init__()
        self.wav2vec_model = model
        self.quantizer = quantizer
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.num_negatives = num_negatives
        
        # A learned vector that replaces the features at masked timesteps
        self.mask_embedding = nn.Parameter(torch.FloatTensor(model.encoder.proj_linear.in_features).uniform_())

        # A final projection from the Transformer output to the quantization dimension
        # for the contrastive loss calculation.
        self.context_projection = nn.Linear(model.encoder.proj_linear.out_features, quantizer.input_dim)

    def _apply_masking(self, features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Applies the time-masking strategy from the Wav2Vec 2.0 paper."""
        batch_size, seq_len, feat_dim = features.shape
        
        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)
            
        # 1. Sample start indices for masking spans
        num_to_mask = int(math.ceil(seq_len * self.mask_prob))
        mask_start_indices = torch.randint(0, seq_len - self.mask_length + 1, (batch_size, num_to_mask), device=features.device)
        
        # 2. Create the final boolean mask by setting spans to True
        time_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)
        for i in range(batch_size):
            for start in mask_start_indices[i]:
                time_mask[i, start:start + self.mask_length] = True
        
        # Ensure that we do not mask padded positions
        time_mask = time_mask & ~padding_mask
        
        # 3. Apply the mask by replacing features with the learned mask embedding
        masked_features = features.clone()
        masked_features[time_mask] = self.mask_embedding
        
        return masked_features, time_mask

    def forward(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None, tau: float = 1.0):
        # 1. Get latent features from the convolutional feature extractor
        latent_features = self.wav2vec_model.feature_extractor(waveforms)
        
        # Calculate padding mask from lengths for the feature space
        padding_mask = None
        output_lengths = None
        if lengths is not None:
            output_lengths = self.wav2vec_model._get_output_lengths(lengths)
            max_len = latent_features.size(1)
            padding_mask = torch.arange(max_len, device=waveforms.device)[None, :] >= output_lengths[:, None]
            
        # 2. Quantize the *unmasked* latent features to get the targets (q_t)
        quantized_targets, perplexity = self.quantizer(latent_features, tau=tau)
        
        # 3. Apply masking to the latent features to create the Transformer input
        masked_features, time_mask = self._apply_masking(latent_features, padding_mask)
        
        # 4. Get contextualized representations (c_t) from the Transformer
        context_vectors = self.wav2vec_model.encoder(masked_features, lengths=output_lengths)
        
        # 5. Select only the values at masked timesteps for loss calculation
        context_vectors_masked = context_vectors[time_mask]
        quantized_targets_masked = quantized_targets[time_mask]

        # Project context vectors to match the quantization dimension
        context_vectors_masked = self.context_projection(context_vectors_masked)
        
        # 6. Sample negative examples from other masked steps in the same batch
        # This is a simplified but effective implementation of negative sampling.
        num_masked_total = context_vectors_masked.size(0)
        # Create random indices to sample from the masked targets
        negative_indices = torch.randint(0, num_masked_total, (num_masked_total, self.num_negatives))
        negatives = quantized_targets_masked[negative_indices]
        
        # Return all components needed for loss calculation
        return context_vectors_masked, quantized_targets_masked, negatives, perplexity

# ==============================================================================
# 3. Model Factory and Main Execution Block
# ==============================================================================

def wav2vec2_base_pretrain_model():
    """
    A factory function that builds the complete pre-training model with the 'base' configuration.
    """
    # Build base model architecture (hyperparameters from the paper)
    conv_layer_configs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    feature_extractor = FeatureExtractor(conv_layer_configs)
    encoder = TransformerEncoder(
        in_features=512, embed_dim=768, proj_dropout=0.1, pos_conv_kernel=128,
        pos_conv_groups=16, num_layers=12, num_heads=12, ffn_dim=3072,
        dropout=0.1, attention_dropout=0.1, layer_drop=0.1,
    )
    model = Wav2Vec2Model(feature_extractor, encoder)
    
    # Build quantizer (hyperparameters from the paper)
    quantizer = Quantizer(input_dim=512, num_groups=2, num_vars=320)
    
    # Wrap them in the pre-training model which handles masking and loss preparation
    return Wav2Vec2PretrainModel(model, quantizer)


if __name__ == "__main__":
    # This block demonstrates how to build the model and run a single pre-training step.
    # It's useful for debugging and verifying the model architecture and data flow.
    
    print("--- Building Wav2Vec2-base pre-training model ---")
    pretrain_model = wav2vec2_base_pretrain_model()
    print(f"Total model parameters: {sum(p.numel() for p in pretrain_model.parameters()) / 1e6:.2f}M")

    # --- Setup for a demonstration training step ---
    contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
    diversity_loss_fn = DiversityLoss()
    alpha = 0.1  # Weight for diversity loss as in the paper
    
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=5e-4)

    print("\n--- Running a single pre-training step ---")
    
    # Create a dummy batch of audio data
    batch_size = 4
    audio_len_seconds = 5
    sample_rate = 16000
    dummy_waveforms = torch.randn(batch_size, audio_len_seconds * sample_rate)
    # Create dummy lengths to simulate a batch with variable-length audio
    dummy_lengths = torch.tensor([sample_rate * 5, sample_rate * 4, sample_rate * 5, sample_rate * 3])

    # --- Training Loop (single step demonstration) ---
    pretrain_model.train()
    optimizer.zero_grad()
    
    # Forward pass to get outputs for loss calculation
    c_t, q_t, negatives, perplexity = pretrain_model(dummy_waveforms, lengths=dummy_lengths)
    
    # Loss calculation
    contrastive_loss = contrastive_loss_fn(c_t, q_t, negatives)
    diversity_loss = diversity_loss_fn(perplexity)
    
    total_loss = contrastive_loss + alpha * diversity_loss
    
    # Backward pass and optimization step
    total_loss.backward()
    optimizer.step()
    
    print(f"Input waveforms shape: {dummy_waveforms.shape}")
    print(f"Context vectors (c_t) shape: {c_t.shape} (Total masked steps, Feature Dim)")
    print(f"Quantized targets (q_t) shape: {q_t.shape}")
    print(f"Negative samples shape: {negatives.shape} (Total masked steps, Num Negatives, Feature Dim)")
    print("-" * 20)
    print(f"Contrastive Loss (L_m): {contrastive_loss.item():.4f}")
    print(f"Codebook Perplexity: {perplexity.item():.4f}")
    print(f"Diversity Loss (L_d): {diversity_loss.item():.4f}")
    print(f"Total Combined Loss: {total_loss.item():.4f}")
    print("\nPre-training step completed successfully!")
