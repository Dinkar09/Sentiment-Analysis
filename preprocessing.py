"""
preprocessing.py
----------------
Loads 'sentiment_data.csv', cleans the 'text' column, tokenizes it with
BertTokenizer, prints output tensors ready for the custom BERT model
defined in train.py, and saves the BERT-ready tensors to JSON inside
the 'Preprocessed data/' subdirectory.

Expected BERT input shape : [batch_size, sequence_length=128]
Output file               : Preprocessed data/bert_tokenised_output.json
"""

import re
import os
import json
import pandas as pd
import torch
from datetime import datetime
from transformers import BertTokenizer

# ─────────────────────────────────────────────
# STEP 1 ─ Load CSV
# ─────────────────────────────────────────────
CSV_PATH    = "sentiment_data.csv"
TEXT_COL    = "text"
MAX_LENGTH  = 128          # Fixed sequence length expected by the BERT model

df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded '{CSV_PATH}'  →  {len(df)} rows, columns: {df.columns.tolist()}")


# ─────────────────────────────────────────────
# STEP 2 ─ Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the preprocessing pipeline used in Sentiment_Analysis.ipynb:

    1. Lowercase                  – normalise case for consistent tokenisation
    2. Remove URLs                – http / https / www links add no sentiment signal
    3. Remove HTML tags           – strip any markup like <br>, &amp; etc.
    4. Remove @mentions           – Twitter handles are not informative
    5. Remove #hashtag symbols    – keep the word, drop the '#' prefix
    6. Keep only alphanumeric +
       essential punctuation      – strip most special chars; keep .,!?' so
                                    BERT sub-word tokeniser can handle contractions
    7. Collapse extra whitespace  – multiple spaces → single space, strip ends
    """

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs  (http/https/ftp/www)
    text = re.sub(r"http\S+|https\S+|ftp\S+|www\.\S+", "", text)

    # 3. Remove HTML tags and HTML entities
    text = re.sub(r"<[^>]+>", "", text)          # tags like <br>
    text = re.sub(r"&[a-z]+;", " ", text)         # entities like &amp;

    # 4. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 5. Remove '#' before hashtag words (keep the word itself)
    text = re.sub(r"#(\w+)", r"\1", text)

    # 6. Remove characters that are NOT letters, digits, spaces,
    #    or the core punctuation that BERT handles well: . , ! ? ' -
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", " ", text)

    # 7. Collapse whitespace / newlines / tabs
    text = re.sub(r"\s+", " ", text).strip()

    return text


raw_texts     = df[TEXT_COL].tolist()
cleaned_texts = [clean_text(t) for t in raw_texts]

# ── Quick sanity check ──────────────────────────────────────────────────────
print("\n─── Text Cleaning Sample (first 5 rows) ─────────────────────────────")
for i in range(5):
    print(f"  BEFORE : {repr(raw_texts[i])}")
    print(f"  AFTER  : {repr(cleaned_texts[i])}")
    print()


# ─────────────────────────────────────────────
# STEP 3 ─ BertTokenizer
# ─────────────────────────────────────────────
print("Loading BertTokenizer ('bert-base-uncased') …")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✓ Tokenizer loaded\n")

# tokenizer() with padding + truncation produces tensors of shape
# [batch_size, MAX_LENGTH] — exactly what the custom BERT model in
# train.py expects for input_ids and attention_mask.
encoded = tokenizer(
    cleaned_texts,
    padding      = "max_length",   # pad every sequence to MAX_LENGTH
    truncation   = True,           # truncate sequences longer than MAX_LENGTH
    max_length   = MAX_LENGTH,
    return_tensors = "pt",         # return PyTorch tensors
)

input_ids      : torch.Tensor = encoded["input_ids"]        # [N, 128]
attention_mask : torch.Tensor = encoded["attention_mask"]   # [N, 128]
token_type_ids : torch.Tensor = encoded["token_type_ids"]   # [N, 128]  ← segment_ids


# ─────────────────────────────────────────────
# STEP 4 ─ Print Results
# ─────────────────────────────────────────────
print("=" * 60)
print("BERT-READY TOKENISED OUTPUT")
print("=" * 60)

print(f"\n  Total sentences     : {input_ids.shape[0]}")
print(f"  Sequence length     : {input_ids.shape[1]}  (fixed = {MAX_LENGTH})")
print(f"\n  input_ids      shape: {list(input_ids.shape)}")
print(f"  attention_mask shape: {list(attention_mask.shape)}")
print(f"  token_type_ids shape: {list(token_type_ids.shape)}")

print("\n─── First 3 rows of input_ids ─────────────────────────────────")
for i in range(3):
    print(f"  [{i}] {input_ids[i, :20].tolist()} …")

print("\n─── First 3 rows of attention_mask ────────────────────────────")
for i in range(3):
    print(f"  [{i}] {attention_mask[i, :20].tolist()} …")

print("\n─── Vocabulary mapping for first sentence (first 10 tokens) ───")
first_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
print(f"  Tokens : {first_tokens[:10]}")
print(f"  IDs    : {input_ids[0, :10].tolist()}")

print("\n" + "=" * 60)
print("HOW TO FEED INTO YOUR BERT MODEL (train.py)")
print("=" * 60)
print("""
  # Build the attention mask in the 4-D shape expected by MultiHeadAttention:
  mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 128]

  # Forward pass (segment_ids = token_type_ids, all zeros for single-sentence input)
  sequence_output, pooled_output = bert_model(
      input_ids      = input_ids,        # [N, 128]
      segment_ids    = token_type_ids,   # [N, 128]
      mask           = mask_4d,          # [N, 1, 1, 128]
  )
  # sequence_output : [N, 128, 768]
  # pooled_output   : [N, 768]
""")


# ─────────────────────────────────────────────
# STEP 5 ─ Save Tokenised Output to JSON
# ─────────────────────────────────────────────
OUTPUT_DIR  = "Preprocessed data"
OUTPUT_FILE = "bert_tokenised_output.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_payload = {
    "metadata": {
        "timestamp"         : datetime.now().isoformat(),
        "source_file"       : CSV_PATH,
        "total_sentences"   : input_ids.shape[0],
        "sequence_length"   : input_ids.shape[1],
        "tokenizer"         : "bert-base-uncased",
        "tensor_shapes": {
            "input_ids"      : list(input_ids.shape),
            "attention_mask" : list(attention_mask.shape),
            "token_type_ids" : list(token_type_ids.shape),
        },
        "bert_model_usage": {
            "input_ids"      : "input_ids      → bert_model(input_ids, ...)",
            "token_type_ids" : "token_type_ids → bert_model(..., segment_ids, ...)",
            "attention_mask" : "attention_mask.unsqueeze(1).unsqueeze(2) → [N,1,1,128] mask",
        },
    },
    "data": {
        # Each tensor stored as a 2-D list: [ [128 ints], [128 ints], ... ]
        "input_ids"      : input_ids.tolist(),
        "attention_mask" : attention_mask.tolist(),
        "token_type_ids" : token_type_ids.tolist(),
    },
}

save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print("=" * 60)
print("SAVED: BERT-READY TOKENISED OUTPUT")
print("=" * 60)
print(f"  Directory : {OUTPUT_DIR}/")
print(f"  File      : {OUTPUT_FILE}")
print(f"  Full path : {save_path}")
print(f"  Keys      : metadata  →  source, shapes, timestamps")
print(f"            : data      →  input_ids, attention_mask, token_type_ids")
print("=" * 60)