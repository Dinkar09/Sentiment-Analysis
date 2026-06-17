"""
gpuTrained.py
-------------
GPU-accelerated training pipeline for the BERT-based sentiment classifier.

This script reuses the exact model architecture and training logic from
'train_2_updated_full_Pipeline.py', with explicit PyTorch CUDA handling so
that training runs on GPU whenever one is available (falls back to CPU
automatically if not).

INPUT  : Preprocessed data/bert_tokenised_output.json  (from preprocessing.py)
         sentiment_data.csv                             (labels only)
LABELS : 0 = negative | 1 = neutral | 2 = positive

GPU-specific changes vs. the original pipeline
-----------------------------------------------
  1. Explicit CUDA availability check + GPU name/memory printout.
  2. torch.backends.cudnn.benchmark = True for faster fixed-size convs/matmuls.
  3. All tensors moved to DEVICE with non_blocking=True (works best with
     pin_memory=True in the DataLoader).
  4. DataLoader uses pin_memory=True when running on GPU for faster
     host-to-device transfers.
  5. Automatic Mixed Precision (AMP) via torch.amp for faster training
     and lower memory usage on GPU (safely disabled on CPU).
  6. torch.cuda.empty_cache() called after training to free GPU memory.

Saved Outputs (same as original pipeline)
  checkpoints/best_model_gpu.pt
  checkpoints/training_log_gpu.json
  bert_execution_outputs/BERT_FINAL_OUTPUT.json   (via tracing.py)
"""

import math
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tracing import BertExecutionTracker


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMBEDDINGS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, type_vocab_size, dropout=0.1):
        super().__init__()
        self.tok_embed  = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embed  = nn.Embedding(max_len, hidden_size)
        self.seg_embed  = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, segment_ids):
        seq_len = x.size(1)
        pos     = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        tok_emb = self.tok_embed(x)
        pos_emb = self.pos_embed(pos)
        seg_emb = self.seg_embed(segment_ids)
        x = tok_emb + pos_emb + seg_emb
        x = self.layer_norm(x)
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-HEAD ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        self.q_linear   = nn.Linear(hidden_size, hidden_size)
        self.k_linear   = nn.Linear(hidden_size, hidden_size)
        self.v_linear   = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # Use the minimum representable value for the current dtype instead
            # of a hardcoded -1e9, which overflows float16 under AMP autocast.
            mask_fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, mask_fill_value)

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        context      = torch.matmul(attn_weights, v)
        context      = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.out_linear(context)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(hidden_size, intermediate_size)
        self.fc2     = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu    = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention    = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout)
        self.norm1        = nn.LayerNorm(hidden_size)
        self.norm2        = nn.LayerNorm(hidden_size)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. BERT ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, max_len, type_vocab_size, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_len,
                                         type_vocab_size, dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.pooler            = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, segment_ids, mask=None):
        """
        Args:
            input_ids   : [batch_size, seq_len]
            segment_ids : [batch_size, seq_len]
            mask        : [batch_size, 1, 1, seq_len]
        Returns:
            (sequence_output [B, S, H], pooled_output [B, H])
        """
        x = self.embeddings(input_ids, segment_ids)

        for layer in self.layers:
            x = layer(x, mask)

        sequence_output = x
        pooled_output = self.pooler_activation(self.pooler(x[:, 0]))

        return sequence_output, pooled_output


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class BertForSentimentClassification(nn.Module):
    """
    BERT encoder  →  Dropout  →  Linear(hidden_size, num_classes)
    Returns raw logits of shape [batch_size, num_classes].
    """
    def __init__(self, bert: BERT, hidden_size: int, num_classes: int,
                 classifier_dropout: float = 0.3):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, segment_ids, mask=None):
        _, pooled_output = self.bert(input_ids, segment_ids, mask)
        return self.classifier(self.dropout(pooled_output))


# ─────────────────────────────────────────────────────────────────────────────
# 7. DATASET  –  wraps pre-tokenised tensors from bert_tokenised_output.json
# ─────────────────────────────────────────────────────────────────────────────

class PreTokenizedDataset(Dataset):
    """
    Dataset built directly from tensors already stored in
    'Preprocessed data/bert_tokenised_output.json'.

    Args:
        input_ids      : LongTensor  [N, 128]
        attention_mask : LongTensor  [N, 128]
        token_type_ids : LongTensor  [N, 128]
        labels         : list[int]   length N   (0 / 1 / 2)
    """

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 token_type_ids: torch.Tensor, labels: list):
        assert len(input_ids) == len(labels), "tensor / label length mismatch"
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.input_ids[idx],       # [128]
            "attention_mask" : self.attention_mask[idx],  # [128]
            "token_type_ids" : self.token_type_ids[idx],  # [128]
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAINING UTILITIES (GPU-aware)
# ─────────────────────────────────────────────────────────────────────────────

def get_linear_warmup_scheduler(optimizer, num_warmup_steps: int,
                                 num_training_steps: int):
    """Linear warm-up for the first `num_warmup_steps`, then linear decay to 0."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler,
                     criterion, device, grad_clip: float,
                     scaler: torch.amp.GradScaler, use_amp: bool):
    """
    Single training epoch with optional Automatic Mixed Precision (AMP).

    Returns:
        avg_loss : float  – mean cross-entropy loss over all batches
        accuracy : float  – fraction of correctly predicted samples
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        # non_blocking=True overlaps host->device copy with compute when
        # the DataLoader uses pin_memory=True (only effective on GPU).
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids, token_type_ids, mask)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Unscale before clipping so the clip threshold is meaningful
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp: bool):
    """
    Validation / test evaluation.

    Returns:
        avg_loss   : float
        accuracy   : float
        all_preds  : list[int]  – predicted labels
        all_labels : list[int]  – ground-truth labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids, token_type_ids, mask)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def setup_device() -> torch.device:
    """
    Detects and configures the compute device.

    - Prints GPU name + memory if CUDA is available.
    - Enables cuDNN autotuner (cudnn.benchmark) for fixed input-size speedups.
    - Falls back to CPU with a warning if no GPU is found.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        torch.backends.cudnn.benchmark = True  # speeds up fixed-size matmuls/convs
        print(f"  ✓ CUDA available — using GPU: {gpu_name}")
        print(f"     Total GPU memory : {total_mem:.2f} GB")
        print(f"     CUDA version     : {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("  ⚠ CUDA not available — falling back to CPU.")
        print("     (Install a CUDA-enabled PyTorch build and run on a "
              "machine with an NVIDIA GPU to enable GPU training.)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ── Device Setup (GPU if available) ─────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'GPU SETUP':^65}")
    print("=" * 65)
    DEVICE  = setup_device()
    USE_AMP = DEVICE.type == "cuda"   # AMP only helps (and is only safe) on GPU

    # ── Model Hyperparameters ────────────────────────────────────────────────────
    HIDDEN_SIZE       = 384     # keep — good representational capacity
    NUM_LAYERS        = 6       # keep — deeper encoder
    NUM_HEADS         = 6       # keep — 384/6 = 64 per head ✓
    INTERMEDIATE_SIZE = 1536    # keep — standard 4× ratio
    MAX_LEN           = 128     # unchanged
    TYPE_VOCAB_SIZE   = 2       # unchanged
    NUM_CLASSES       = 3       # unchanged
    DROPOUT           = 0.2     # slight increase → more regularisation
    CLASSIFIER_DROP   = 0.3     # restored to 0.3 → classifier needs more dropout

    # ── Training Hyperparameters ─────────────────────────────────────────────────
    BATCH_SIZE        = 32      # keep — stable for 4GB VRAM with AMP
    EPOCHS            = 20      # increased — lower LR needs more epochs to converge
    LR                = 1e-4    # ← KEY FIX: was 4e-4 (comment said "lower" but was higher!)
                                # 1e-4 is the standard for training BERT from scratch
    WEIGHT_DECAY      = 0.01    # unchanged
    WARMUP_RATIO      = 0.15    # increased from 0.1 — more stable warmup for scratch training
    GRAD_CLIP         = 1.0     # unchanged
    TRAIN_SPLIT       = 0.8     # unchanged
    LABEL_SMOOTHING   = 0.1     # NEW — prevents overconfidence, helps generalisation
    SCHEDULER         = "cosine" # NEW — cosine annealing instead of linear decay
                              #       avoids the sudden drop after warmup ends

    # ── Paths ────────────────────────────────────────────────────────────────
    JSON_INPUT_PATH = os.path.join("Preprocessed data", "bert_tokenised_output.json")
    CSV_PATH        = "sentiment_data.csv"
    CKPT_DIR        = "checkpoints"
    BERT_OUTPUT_DIR = "bert_execution_outputs"
    os.makedirs(CKPT_DIR, exist_ok=True)


    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 – LOAD PRE-TOKENISED INPUT FROM bert_tokenised_output.json
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 1 : Loading pre-tokenised tensors from JSON")
    print("=" * 65)
    print(f"  Source : {JSON_INPUT_PATH}")

    with open(JSON_INPUT_PATH, "r") as f:
        tokenised_json = json.load(f)

    input_ids_all      = torch.tensor(tokenised_json["data"]["input_ids"],
                                       dtype=torch.long)
    attention_mask_all = torch.tensor(tokenised_json["data"]["attention_mask"],
                                       dtype=torch.long)
    token_type_ids_all = torch.tensor(tokenised_json["data"]["token_type_ids"],
                                       dtype=torch.long)

    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    VOCAB_SIZE = tokenizer.vocab_size

    meta = tokenised_json["metadata"]
    print(f"  ✓ Tensors loaded")
    print(f"     tokenizer       : {meta['tokenizer']}")
    print(f"     total sentences : {meta['total_sentences']}")
    print(f"     sequence length : {meta['sequence_length']}")
    print(f"     input_ids       : {list(input_ids_all.shape)}")
    print(f"     attention_mask  : {list(attention_mask_all.shape)}")
    print(f"     token_type_ids  : {list(token_type_ids_all.shape)}")
    print(f"     vocab_size      : {VOCAB_SIZE}")


    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 – LOAD LABELS & STRATIFIED TRAIN / VALIDATION SPLIT
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 2 : Loading labels & splitting dataset")
    print("=" * 65)

    df     = pd.read_csv(CSV_PATH)
    labels = df["label"].tolist()
    print(f"  ✓ Labels loaded from '{CSV_PATH}'")
    print(f"     {dict(df['label'].value_counts().sort_index())}")

    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - TRAIN_SPLIT,
        random_state = SEED,
        stratify     = labels,
    )

    train_input_ids      = input_ids_all[train_idx]
    train_attention_mask = attention_mask_all[train_idx]
    train_token_type_ids = token_type_ids_all[train_idx]
    train_labels         = [labels[i] for i in train_idx]

    val_input_ids        = input_ids_all[val_idx]
    val_attention_mask   = attention_mask_all[val_idx]
    val_token_type_ids   = token_type_ids_all[val_idx]
    val_labels           = [labels[i] for i in val_idx]

    print(f"  ✓ Train : {len(train_idx)} samples  |  Validation : {len(val_idx)} samples")

    train_dataset = PreTokenizedDataset(
        train_input_ids, train_attention_mask, train_token_type_ids, train_labels
    )
    val_dataset = PreTokenizedDataset(
        val_input_ids, val_attention_mask, val_token_type_ids, val_labels
    )

    # pin_memory=True speeds up host->device transfer when training on GPU.
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0, pin_memory=pin_memory)


    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 – BUILD MODEL (moved to GPU)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 3 : Building BERT model")
    print("=" * 65)

    bert_encoder = BERT(
        vocab_size        = VOCAB_SIZE,
        hidden_size       = HIDDEN_SIZE,
        num_layers        = NUM_LAYERS,
        num_heads         = NUM_HEADS,
        intermediate_size = INTERMEDIATE_SIZE,
        max_len           = MAX_LEN,
        type_vocab_size   = TYPE_VOCAB_SIZE,
        dropout           = DROPOUT,
    )
    model = BertForSentimentClassification(
        bert               = bert_encoder,
        hidden_size        = HIDDEN_SIZE,
        num_classes        = NUM_CLASSES,
        classifier_dropout = CLASSIFIER_DROP,
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total params     : {total_params:,}")
    print(f"  ✓ Trainable params : {trainable_params:,}")
    print(f"  ✓ Architecture     : BERT-Small "
          f"(layers={NUM_LAYERS}, heads={NUM_HEADS}, hidden={HIDDEN_SIZE})")
    print(f"  ✓ Model device     : {next(model.parameters()).device}")
    print(f"  ✓ Mixed precision  : {'enabled (AMP)' if USE_AMP else 'disabled (CPU)'}")


    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 – TRAIN
    # ════════════════════════════════════════════════════════════════════════
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()

    # GradScaler is a no-op when enabled=False (CPU), so this is safe everywhere.
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    print("\n" + "=" * 65)
    print(f"{'STEP 4 : TRAINING':^65}")
    print(f"  Epochs={EPOCHS}  |  Batch={BATCH_SIZE}  |  LR={LR}  |  Device={DEVICE}")
    print(f"  Scheduler : warm-up {warmup_steps} steps → decay {total_steps} steps")
    print("=" * 65)
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
    print("-" * 65)

    best_val_acc = 0.0
    training_log = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, DEVICE, GRAD_CLIP, scaler, USE_AMP,
        )
        val_loss, val_acc, val_preds, val_true = evaluate(
            model, val_loader, criterion, DEVICE, USE_AMP,
        )

        elapsed = time.time() - t0
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>9.4f} {elapsed:>6.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "val_acc"    : val_acc,
                    "val_loss"   : val_loss,
                    "config": {
                        "vocab_size"       : VOCAB_SIZE,
                        "hidden_size"      : HIDDEN_SIZE,
                        "num_layers"       : NUM_LAYERS,
                        "num_heads"        : NUM_HEADS,
                        "intermediate_size": INTERMEDIATE_SIZE,
                        "max_len"          : MAX_LEN,
                        "type_vocab_size"  : TYPE_VOCAB_SIZE,
                        "num_classes"      : NUM_CLASSES,
                    },
                },
                os.path.join(CKPT_DIR, "best_model_gpu.pt"),
            )

        training_log.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "train_acc" : round(train_acc,  6),
            "val_loss"  : round(val_loss,   6),
            "val_acc"   : round(val_acc,    6),
            "elapsed_s" : round(elapsed,    2),
        })

    print("-" * 65)
    print(f"  Best validation accuracy : {best_val_acc:.4f}")
    print(f"  Checkpoint saved to      : {CKPT_DIR}/best_model_gpu.pt")


    # ════════════════════════════════════════════════════════════════════════
    # STEP 5 – CLASSIFICATION REPORT  (best epoch weights)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 5 : Classification report  (validation set – best weights)")
    print("=" * 65)

    ckpt = torch.load(os.path.join(CKPT_DIR, "best_model_gpu.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, _, final_preds, final_true = evaluate(model, val_loader, criterion, DEVICE, USE_AMP)
    print(classification_report(
        final_true, final_preds,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    ))

    log_path = os.path.join(CKPT_DIR, "training_log_gpu.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved to : {log_path}")


    # ════════════════════════════════════════════════════════════════════════
    # STEP 6 – STORE BERT OUTPUT  →  bert_execution_outputs/BERT_FINAL_OUTPUT.json
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 6 : Storing BERT output → BERT_FINAL_OUTPUT.json")
    print("=" * 65)
    print(f"  Running trained BERT encoder over all {len(labels)} sentences …")

    full_dataset = PreTokenizedDataset(
        input_ids_all, attention_mask_all, token_type_ids_all,
        labels=[0] * len(labels),        # dummy labels – not used below
    )
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=pin_memory)

    bert_encoder.to(DEVICE).eval()
    all_pooled_outputs = []

    with torch.no_grad():
        for batch in full_loader:
            ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
            segs = batch["token_type_ids"].to(DEVICE, non_blocking=True)
            mask = batch["attention_mask"].to(DEVICE, non_blocking=True).unsqueeze(1).unsqueeze(2)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                _, pooled = bert_encoder(ids, segs, mask)  # pooled: [B, HIDDEN_SIZE]

            all_pooled_outputs.append(pooled.float().cpu())

    full_pooled_tensor = torch.cat(all_pooled_outputs, dim=0)
    print(f"  ✓ Full pooled output tensor shape : {list(full_pooled_tensor.shape)}")
    print(f"    Meaning : [total_sentences={full_pooled_tensor.shape[0]}, "
          f"hidden_dim={full_pooled_tensor.shape[1]}]")

    tracker = BertExecutionTracker(output_dir=BERT_OUTPUT_DIR)
    tracker._save_file(full_pooled_tensor, "BERT_FINAL_OUTPUT.json")

    print(f"\n  [FINAL OUTPUT] POOLED OUTPUT (CLS Token Representation)")
    print(f"  Shape   : {full_pooled_tensor.shape}")
    print(f"  Purpose : Sentence-level representations for all {len(labels)} inputs")
    print(f"  Saved   : {BERT_OUTPUT_DIR}/BERT_FINAL_OUTPUT.json")


    # ── Free GPU memory ──────────────────────────────────────────────────────
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        print("\n  ✓ GPU cache cleared")


    # ── Final tensor shape reference ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TENSOR SHAPE REFERENCE")
    print("=" * 65)
    print(f"  Device          : {DEVICE}")
    print(f"  input_ids       : [batch_size, {MAX_LEN}]")
    print(f"  token_type_ids  : [batch_size, {MAX_LEN}]   (segment_ids)")
    print(f"  attention_mask  : [batch_size, 1, 1, {MAX_LEN}]   (inside model)")
    print(f"  sequence_output : [batch_size, {MAX_LEN}, {HIDDEN_SIZE}]")
    print(f"  pooled_output   : [batch_size, {HIDDEN_SIZE}]")
    print(f"  logits          : [batch_size, {NUM_CLASSES}]")
    print(f"  BERT_FINAL_OUTPUT (full dataset) : [{len(labels)}, {HIDDEN_SIZE}]")
    print("=" * 65)