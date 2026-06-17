"""
train.py
--------
Full training pipeline for a BERT-based sentiment classifier.

INPUT  : Preprocessed data/bert_tokenised_output.json  (from preprocessing.py)
         sentiment_data.csv                             (labels only)
LABELS : 0 = negative | 1 = neutral | 2 = positive

Execution Flow
──────────────
  STEP 1  Load pre-tokenised tensors from bert_tokenised_output.json
  STEP 2  Load labels from sentiment_data.csv & stratified train/val split
  STEP 3  Build BERT encoder + classification head
  STEP 4  Train for EPOCHS with AdamW + linear warm-up scheduler
  STEP 5  Evaluate on validation set; save best checkpoint
  STEP 6  Run full dataset through trained BERT encoder → save pooled output
          to  bert_execution_outputs/BERT_FINAL_OUTPUT.json  (via tracing.py)

Architecture Config  (BERT-Small – practical for training from scratch)
  VOCAB_SIZE       = auto-detected from tokenizer
  HIDDEN_SIZE      = 256
  NUM_LAYERS       = 4
  NUM_HEADS        = 4
  INTERMEDIATE     = 512
  MAX_LEN          = 128
  TYPE_VOCAB_SIZE  = 2
  NUM_CLASSES      = 3

Training Config
  BATCH_SIZE   = 32
  EPOCHS       = 5
  LR           = 2e-4
  WEIGHT_DECAY = 0.01
  WARMUP_RATIO = 0.1
  GRAD_CLIP    = 1.0
  TRAIN_SPLIT  = 0.8

Saved Outputs
  checkpoints/best_model.pt                       – best val-accuracy weights
  checkpoints/training_log.json                   – per-epoch metrics
  bert_execution_outputs/BERT_FINAL_OUTPUT.json   – pooled BERT output (tracing.py)
"""

import math
import re
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMBEDDINGS MODULE  (unchanged)
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
# 2. MULTI-HEAD ATTENTION  (unchanged)
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
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        context      = torch.matmul(attn_weights, v)
        context      = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.out_linear(context)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEED-FORWARD NETWORK  (unchanged)
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
# 4. TRANSFORMER BLOCK  (unchanged)
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
# 5. BERT ENCODER  (unchanged)
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
        # STEP 1 – Input embeddings
        x = self.embeddings(input_ids, segment_ids)

        # STEP 2 – Encoder stack
        for layer in self.layers:
            x = layer(x, mask)

        # STEP 3 – Sequence output (all token hidden states)
        sequence_output = x

        # STEP 4 – Pooled [CLS] output for classification
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
        """
        Args:
            input_ids   : [B, 128]
            segment_ids : [B, 128]
            mask        : [B, 1, 1, 128]
        Returns:
            logits : [B, num_classes]
        """
        _, pooled_output = self.bert(input_ids, segment_ids, mask)
        return self.classifier(self.dropout(pooled_output))


# ─────────────────────────────────────────────────────────────────────────────
# 7. DATASET  –  wraps pre-tokenised tensors from bert_tokenised_output.json
# ─────────────────────────────────────────────────────────────────────────────

class PreTokenizedDataset(Dataset):
    """
    Dataset built directly from tensors already stored in
    'Preprocessed data/bert_tokenised_output.json'.

    No re-tokenisation is done here; every sample is a slice of the
    pre-built [N, 128] tensors produced by preprocessing.py.

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
# 8. TRAINING UTILITIES
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
                    criterion, device, grad_clip: float):
    """
    Single training epoch.

    Returns:
        avg_loss : float  – mean cross-entropy loss over all batches
        accuracy : float  – fraction of correctly predicted samples
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)        # [B, 128]
        token_type_ids = batch["token_type_ids"].to(device)   # [B, 128]
        attention_mask = batch["attention_mask"].to(device)   # [B, 128]
        labels         = batch["label"].to(device)            # [B]

        # Reshape mask → [B, 1, 1, 128] for MultiHeadAttention
        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        optimizer.zero_grad()
        logits = model(input_ids, token_type_ids, mask)       # [B, num_classes]
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
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
        input_ids      = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        mask   = attention_mask.unsqueeze(1).unsqueeze(2)
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
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tracing import BertExecutionTracker

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED = 42
    torch.manual_seed(SEED)

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ── Model Hyperparameters ────────────────────────────────────────────────
    HIDDEN_SIZE       = 256
    NUM_LAYERS        = 4
    NUM_HEADS         = 4
    INTERMEDIATE_SIZE = 512
    MAX_LEN           = 128
    TYPE_VOCAB_SIZE   = 2
    NUM_CLASSES       = 3       # negative=0 | neutral=1 | positive=2
    DROPOUT           = 0.1
    CLASSIFIER_DROP   = 0.3

    # ── Training Hyperparameters ─────────────────────────────────────────────
    BATCH_SIZE   = 32
    EPOCHS       = 5
    LR           = 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1          # 10 % of total steps for linear warm-up
    GRAD_CLIP    = 1.0
    TRAIN_SPLIT  = 0.8

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

    # Reconstruct tensors from the 2-D lists stored by preprocessing.py
    # Shape of each tensor: [total_sentences, 128]
    input_ids_all      = torch.tensor(tokenised_json["data"]["input_ids"],
                                       dtype=torch.long)
    attention_mask_all = torch.tensor(tokenised_json["data"]["attention_mask"],
                                       dtype=torch.long)
    token_type_ids_all = torch.tensor(tokenised_json["data"]["token_type_ids"],
                                       dtype=torch.long)

    # VOCAB_SIZE is derived from the tokenizer (must match what preprocessing.py used)
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
    labels = df["label"].tolist()   # 0 / 1 / 2  (31 232 entries)
    print(f"  ✓ Labels loaded from '{CSV_PATH}'")
    print(f"     {dict(df['label'].value_counts().sort_index())}")

    # Split by index so tensors and labels stay aligned
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - TRAIN_SPLIT,
        random_state = SEED,
        stratify     = labels,
    )

    # Slice pre-tokenised tensors
    train_input_ids      = input_ids_all[train_idx]
    train_attention_mask = attention_mask_all[train_idx]
    train_token_type_ids = token_type_ids_all[train_idx]
    train_labels         = [labels[i] for i in train_idx]

    val_input_ids        = input_ids_all[val_idx]
    val_attention_mask   = attention_mask_all[val_idx]
    val_token_type_ids   = token_type_ids_all[val_idx]
    val_labels           = [labels[i] for i in val_idx]

    print(f"  ✓ Train : {len(train_idx)} samples  |  Validation : {len(val_idx)} samples")

    # Build datasets from pre-tokenised tensors (no re-tokenisation)
    train_dataset = PreTokenizedDataset(
        train_input_ids, train_attention_mask, train_token_type_ids, train_labels
    )
    val_dataset = PreTokenizedDataset(
        val_input_ids, val_attention_mask, val_token_type_ids, val_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)


    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 – BUILD MODEL
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


    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 – TRAIN
    # ════════════════════════════════════════════════════════════════════════
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()

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
            criterion, DEVICE, GRAD_CLIP,
        )
        val_loss, val_acc, val_preds, val_true = evaluate(
            model, val_loader, criterion, DEVICE,
        )

        elapsed = time.time() - t0
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>9.4f} {elapsed:>6.1f}s")

        # Save best checkpoint
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
                os.path.join(CKPT_DIR, "best_model.pt"),
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
    print(f"  Checkpoint saved to      : {CKPT_DIR}/best_model.pt")


    # ════════════════════════════════════════════════════════════════════════
    # STEP 5 – CLASSIFICATION REPORT  (best epoch weights)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 5 : Classification report  (validation set – best weights)")
    print("=" * 65)

    ckpt = torch.load(os.path.join(CKPT_DIR, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, _, final_preds, final_true = evaluate(model, val_loader, criterion, DEVICE)
    print(classification_report(
        final_true, final_preds,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    ))

    # Save training log
    log_path = os.path.join(CKPT_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved to : {log_path}")


    # ════════════════════════════════════════════════════════════════════════
    # STEP 6 – STORE BERT OUTPUT  →  bert_execution_outputs/BERT_FINAL_OUTPUT.json
    #
    # The trained BERT encoder is run over ALL tokenised sentences loaded from
    # bert_tokenised_output.json in mini-batches to stay memory-efficient.
    # All [B, HIDDEN_SIZE] pooled outputs are concatenated into a single
    # tensor [N, HIDDEN_SIZE] which is then saved by BertExecutionTracker
    # (tracing.py) to  bert_execution_outputs/BERT_FINAL_OUTPUT.json,
    # using the same  _save_file  method it uses for every other output.
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 6 : Storing BERT output → BERT_FINAL_OUTPUT.json")
    print("=" * 65)
    print(f"  Running trained BERT encoder over all {len(labels)} sentences …")

    # Build a DataLoader over the FULL dataset (no labels needed here)
    full_dataset = PreTokenizedDataset(
        input_ids_all, attention_mask_all, token_type_ids_all,
        labels=[0] * len(labels),        # dummy labels – not used below
    )
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    bert_encoder.to(DEVICE).eval()
    all_pooled_outputs = []

    with torch.no_grad():
        for batch in full_loader:
            ids  = batch["input_ids"].to(DEVICE)       # [B, 128]
            segs = batch["token_type_ids"].to(DEVICE)  # [B, 128]
            mask = batch["attention_mask"].to(DEVICE).unsqueeze(1).unsqueeze(2)

            # bert_encoder returns (sequence_output, pooled_output)
            _, pooled = bert_encoder(ids, segs, mask)  # pooled: [B, HIDDEN_SIZE]
            all_pooled_outputs.append(pooled.cpu())

    # Concatenate all batches → single tensor [N, HIDDEN_SIZE]
    full_pooled_tensor = torch.cat(all_pooled_outputs, dim=0)
    print(f"  ✓ Full pooled output tensor shape : {list(full_pooled_tensor.shape)}")
    print(f"    Meaning : [total_sentences={full_pooled_tensor.shape[0]}, "
          f"hidden_dim={full_pooled_tensor.shape[1]}]")

    # ── Save via BertExecutionTracker (exactly as tracing.py defines) ────────
    #
    #   _save_file builds:
    #     {
    #       "timestamp"    : <iso string>,
    #       "tensor_shape" : "<shape>",
    #       "data"         : [[...], ...]   ← 2-D list via _tensor_to_json_serializable
    #     }
    #   and writes it to  bert_execution_outputs/BERT_FINAL_OUTPUT.json
    #
    tracker = BertExecutionTracker(output_dir=BERT_OUTPUT_DIR)
    tracker._save_file(full_pooled_tensor, "BERT_FINAL_OUTPUT.json")

    print(f"\n  [FINAL OUTPUT] POOLED OUTPUT (CLS Token Representation)")
    print(f"  Shape   : {full_pooled_tensor.shape}")
    print(f"  Purpose : Sentence-level representations for all {len(labels)} inputs")
    print(f"  Saved   : {BERT_OUTPUT_DIR}/BERT_FINAL_OUTPUT.json")


    # ── Final tensor shape reference ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TENSOR SHAPE REFERENCE")
    print("=" * 65)
    print(f"  input_ids       : [batch_size, {MAX_LEN}]")
    print(f"  token_type_ids  : [batch_size, {MAX_LEN}]   (segment_ids)")
    print(f"  attention_mask  : [batch_size, 1, 1, {MAX_LEN}]   (inside model)")
    print(f"  sequence_output : [batch_size, {MAX_LEN}, {HIDDEN_SIZE}]")
    print(f"  pooled_output   : [batch_size, {HIDDEN_SIZE}]")
    print(f"  logits          : [batch_size, {NUM_CLASSES}]")
    print(f"  BERT_FINAL_OUTPUT (full dataset) : [{len(labels)}, {HIDDEN_SIZE}]")
    print("=" * 65)
