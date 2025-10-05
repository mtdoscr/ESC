import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging
import warnings
warnings.filterwarnings("ignore") # Suppress warnings
import datetime
from torch.utils.tensorboard import SummaryWriter
from Wav2vec2_model import wav2vec2_base_pretrain_model, ContrastiveLoss, DiversityLoss
from dataset import UrbanSound8KDataset

# ==============================================================================
# This script handles the self-supervised pre-training of the Wav2Vec2 model
# on the UrbanSound8K dataset. The goal is to learn useful audio representations
# without using any labels, which can then be used for downstream tasks like classification.
# ==============================================================================


# ==============================================================================
# 1. Configuration
# ==============================================================================
# --- Training Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 16 # Adjust based on available GPU/XPU memory
LEARNING_RATE = 5e-5 # Learning rate for the Adam optimizer

# --- Loss Configuration ---
# Weight for the diversity loss component. The total loss is: L = L_contrastive + alpha * L_diversity
ALPHA_DIVERSITY = 0.005

# --- Dataset Configuration ---
# Using folds 1-9 for training. Fold 10 is held out for validation.
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_SAMPLE_RATE = 16000 # Sample rate the model expects (Hz)
AUDIO_DURATION_SECS = 4 # Duration of audio clips to use for training (in seconds)

# --- Paths ---
DATASET_PATH = "UrbanSound8K"
METADATA_PATH = os.path.join(DATASET_PATH, "UrbanSound8K.csv")
CHECKPOINT_DIR = "checkpoints" # Directory to save model checkpoints
LOG_DIR = "runs" # Directory for TensorBoard logs

# ==============================================================================
# 2. Setup
# ==============================================================================
def setup():
    """
    Initializes the training device, creates necessary directories, and sets up the TensorBoard logger.
    """
    # --- Device Selection ---
    if torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel XPU for training.")
    else:
        device = torch.device("cpu")
        print("XPU not found. Using CPU for training.")

    # --- Directory Creation ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- TensorBoard Logger ---
    # Create a unique log directory for this training run using a timestamp.
    timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_writer = SummaryWriter(os.path.join(LOG_DIR, f"pretrain_{timestamp}"))
    print(f"TensorBoard log directory: {log_writer.log_dir}")
    
    return device, log_writer

# ==============================================================================
# 3. Training Loop
# ==============================================================================
def train(device, log_writer):
    """
    Contains the main pre-training logic, including data loading, the training loop,
    validation, and checkpointing.
    """
    # --- Data Loading ---
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}.")
        print("Please download the UrbanSound8K dataset and place it in the project root.")
        return

    # Create the training dataset and dataloader
    train_dataset = UrbanSound8KDataset(
        annotations_file=METADATA_PATH, 
        audio_dir=DATASET_PATH, 
        folds=TRAIN_FOLDS,
        target_sample_rate=TARGET_SAMPLE_RATE,
        max_duration_seconds=AUDIO_DURATION_SECS
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Loaded {len(train_dataset)} training samples.")

    # Create the validation dataset and dataloader (using fold 10)
    val_dataset = UrbanSound8KDataset(
        annotations_file=METADATA_PATH,
        audio_dir=DATASET_PATH,
        folds=[10], 
        target_sample_rate=TARGET_SAMPLE_RATE,
        max_duration_seconds=AUDIO_DURATION_SECS
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Loaded {len(val_dataset)} validation samples.")


    # --- Model, Optimizer, and Loss Functions ---
    model = wav2vec2_base_pretrain_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    contrastive_loss_fn = ContrastiveLoss(temperature=0.1).to(device)
    diversity_loss_fn = DiversityLoss().to(device)

    # --- Gumbel-Softmax Temperature Annealing ---
    # Linearly anneal the temperature `tau` for the Gumbel-Softmax from a starting value
    # to an ending value over the course of training. This helps in the initial exploration
    # of the codebooks and later stabilization.
    total_steps = EPOCHS * len(train_dataloader)
    start_temp = 1.0
    end_temp = 0.1
    
    # --- Training and Validation ---
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        # --- Training Phase ---
        model.train()
        total_train_loss = 0

        for i, waveforms in enumerate(train_dataloader):
            waveforms = waveforms.to(device)
            optimizer.zero_grad()

            # Calculate the current temperature for this step
            temp_decay_rate = (start_temp - end_temp) / total_steps
            current_temp = max(start_temp - global_step * temp_decay_rate, end_temp)

            # Forward pass through the pre-training model
            # c_t: contextualized vectors from masked positions
            # q_t: quantized target vectors from the same positions (unmasked)
            # negatives: other quantized vectors to be used as negative samples
            # perplexity: a measure of codebook usage diversity
            c_t, q_t, negatives, perplexity = model(waveforms, tau=current_temp)

            # Loss Calculation
            contrastive_loss = contrastive_loss_fn(c_t, q_t, negatives)
            diversity_loss = diversity_loss_fn(perplexity)
            loss = contrastive_loss + ALPHA_DIVERSITY * diversity_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # --- Logging ---
            total_train_loss += loss.item()
            log_writer.add_scalar('Loss/Contrastive_Batch', contrastive_loss.item(), global_step)
            log_writer.add_scalar('Loss/Diversity_Batch', diversity_loss.item(), global_step)
            log_writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            log_writer.add_scalar('Hyperparameters/Gumbel_Temperature', current_temp, global_step)
            global_step += 1

            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}, Temp: {current_temp:.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        log_writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)

        # --- Validation Phase ---
        print(f"--- Validating Epoch {epoch} ---")
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, val_waveforms in enumerate(val_dataloader):
                val_waveforms = val_waveforms.to(device)
                # Use the final temperature for validation
                c_t, q_t, negatives, perplexity = model(val_waveforms, tau=end_temp)
                
                contrastive_loss = contrastive_loss_fn(c_t, q_t, negatives)
                diversity_loss = diversity_loss_fn(perplexity)
                val_loss = contrastive_loss + ALPHA_DIVERSITY * diversity_loss
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        log_writer.add_scalar('Loss/Validation_Epoch', avg_val_loss, epoch)

        print(f"Epoch {epoch} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Validation Loss: {avg_val_loss:.4f}")

        # --- Checkpointing ---
        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'global_step': global_step,
        }

        # 1. Save the latest checkpoint after every epoch
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'pretrain_checkpoint_latest.pth')
        torch.save(checkpoint_payload, latest_checkpoint_path)
        print(f"Saved latest checkpoint to {latest_checkpoint_path}")

        # 2. Save the best checkpoint if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'pretrain_checkpoint_best.pth')
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {avg_val_loss:.4f})")

    log_writer.close()
    print("\n--- Pre-training finished ---")

# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Initialize setup and start the training process
    device, logger = setup()
    train(device, logger)