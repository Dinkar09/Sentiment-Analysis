"""
ui.py
-----
Streamlit Sentiment Analysis Hub.

Loads two trained BERT checkpoints and runs inference from a web UI.

  Model 1  →  checkpoints/best_model.pt      (CPU-trained, BERT-Small 256-4-4)
  Model 2  →  checkpoints/best_model_gpu.pt  (GPU-trained, BERT-Small 384-6-6)

Config (hidden size, layers, heads …) is read DIRECTLY from each checkpoint's
saved "config" dict, so no architecture constant is hard-coded here.

Run:
    streamlit run ui.py
"""

import math
import re
import os

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (Set layout to centered for structural verticality)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Sentiment Analysis Hub",
    page_icon  = "🧠",
    layout     = "centered",
)


# ─────────────────────────────────────────────────────────────────────────────
# BERT ARCHITECTURE CONFIGURATION
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
        x       = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(segment_ids)
        return self.dropout(self.layer_norm(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
        self.q_linear    = nn.Linear(hidden_size, hidden_size)
        self.k_linear    = nn.Linear(hidden_size, hidden_size)
        self.v_linear    = nn.Linear(hidden_size, hidden_size)
        self.out_linear  = nn.Linear(hidden_size, hidden_size)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B = x.size(0)
        q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn    = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.hidden_size)
        return self.out_linear(context)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(hidden_size, intermediate_size)
        self.fc2     = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu    = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention    = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout)
        self.norm1        = nn.LayerNorm(hidden_size)
        self.norm2        = nn.LayerNorm(hidden_size)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


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
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, mask)
        pooled = self.pooler_activation(self.pooler(x[:, 0]))
        return x, pooled


class BertForSentimentClassification(nn.Module):
    def __init__(self, bert: BERT, hidden_size: int, num_classes: int,
                 classifier_dropout: float = 0.3):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, segment_ids, mask=None):
        _, pooled = self.bert(input_ids, segment_ids, mask)
        return self.classifier(self.dropout(pooled))


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION PIPELINE CLEANER
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Seven-step cleaning pipeline (matching preprocessing.py)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|https\S+|ftp\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>",   "", text)
    text = re.sub(r"&[a-z]+;",  " ", text)
    text = re.sub(r"@\w+",      "", text)
    text = re.sub(r"#(\w+)",    r"\1", text)
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+",       " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, inference_device: str):
    device = torch.device(inference_device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg    = ckpt["config"]

    bert_encoder = BERT(
        vocab_size        = cfg["vocab_size"],
        hidden_size       = cfg["hidden_size"],
        num_layers        = cfg["num_layers"],
        num_heads         = cfg["num_heads"],
        intermediate_size = cfg["intermediate_size"],
        max_len           = cfg["max_len"],
        type_vocab_size   = cfg["type_vocab_size"],
    )
    model = BertForSentimentClassification(
        bert        = bert_encoder,
        hidden_size = cfg["hidden_size"],
        num_classes = cfg["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "epoch"   : ckpt.get("epoch",   "—"),
        "val_acc" : ckpt.get("val_acc", 0.0),
        "val_loss": ckpt.get("val_loss", 0.0),
    }
    return model, cfg, meta


@torch.no_grad()
def run_inference(text: str, model, tokenizer, device_str: str, max_len: int = 128):
    LABELS  = ["negative", "neutral", "positive"]
    device  = torch.device(device_str)
    cleaned = clean_text(text)

    encoded = tokenizer(
        cleaned,
        padding        = "max_length",
        truncation     = True,
        max_length     = max_len,
        return_tensors = "pt",
    )

    input_ids      = encoded["input_ids"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    mask           = attention_mask.unsqueeze(1).unsqueeze(2)

    logits = model(input_ids, token_type_ids, mask)
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    label  = LABELS[int(torch.tensor(probs).argmax())]

    return label, probs, cleaned


# ─────────────────────────────────────────────────────────────────────────────
# TRACKING SESSIONS STATES
# ─────────────────────────────────────────────────────────────────────────────

if "selected_model" not in st.session_state:
    st.session_state.selected_model = 1  # Default track: CPU Model

if "result" not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE CONFIG MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    1: {
        "label"      : "CPU trained model",
        "checkpoint" : os.path.join("checkpoints", "best_model.pt"),
        "device"     : "cpu",
    },
    2: {
        "label"      : "GPU trained model",
        "checkpoint" : os.path.join("checkpoints", "best_model_gpu.pt"),
        "device"     : "cuda" if torch.cuda.is_available() else "cpu",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# STRICT VERTICAL COMPONENT STREAM
# ══════════════════════════════════════════════════════════════════════════════

# Fix 1: Header Panel Visibility Fix
st.title("🧠 Sentiment Analysis Hub")
st.markdown("An interactive evaluation interface for optimized BERT classification checkpoints.")
st.markdown("---")

# Fix 2: Only 2 Model Selection Toggles (CPU vs GPU Track)
st.subheader("1. Select Model Track")

btn_cols = st.columns(2)
with btn_cols[0]:
    cpu_lbl = "🟢 CPU trained model (Selected)" if st.session_state.selected_model == 1 else "CPU trained model"
    if st.button(cpu_lbl, key="set_cpu_track", use_container_width=True):
        st.session_state.selected_model = 1
        st.session_state.result = None
        st.rerun()

with btn_cols[1]:
    gpu_lbl = "🟢 GPU trained model (Selected)" if st.session_state.selected_model == 2 else "GPU trained model"
    if st.button(gpu_lbl, key="set_gpu_track", use_container_width=True):
        st.session_state.selected_model = 2
        st.session_state.result = None
        st.rerun()

st.markdown("---")

# Fix 3: Multi-column Split removed to maintain vertical alignment
st.subheader("2. Input Sentence")
user_text = st.text_area(
    label="Enter an English sentence to test sentiment mapping:",
    placeholder="Type here... (e.g., The execution framework performs surprisingly smoothly!)",
    height=120,
    label_visibility="visible"
)

st.markdown("---")

st.subheader("3. Execute Prediction")
run_clicked = st.button("▶ Run Sentiment Analysis", use_container_width=True, type="primary")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOGIC ROUTER
# ─────────────────────────────────────────────────────────────────────────────

if run_clicked:
    if not user_text.strip():
        st.warning("⚠️ Please provide text inside the input zone before attempting analysis.")
    else:
        model_key  = st.session_state.selected_model
        model_meta = MODEL_REGISTRY[model_key]

        if not os.path.exists(model_meta["checkpoint"]):
            st.error(f"❌ Target weights checkpoint missing at: `{model_meta['checkpoint']}`. Train that model segment first.")
        else:
            with st.spinner(f"Computing forward-pass utilizing {model_meta['label']}..."):
                tokenizer = load_tokenizer()
                model, cfg, meta = load_model(model_meta["checkpoint"], model_meta["device"])
                label, probs, cleaned = run_inference(user_text, model, tokenizer, model_meta["device"], max_len=cfg["max_len"])

            st.session_state.result = {
                "label"      : label,
                "probs"      : probs,
                "cleaned"    : cleaned,
                "model_label": model_meta["label"],
                "device"     : model_meta["device"].upper(),
                "cfg"        : cfg,
                "meta"       : meta,
            }
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Clean Native UI Elements for Classification Results
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("4. Classification Results")

res = st.session_state.result

if res is None:
    st.info("🔍 Provide an input string and trigger 'Run Sentiment Analysis' to display live classification metrics.")
else:
    label = res["label"]
    probs = res["probs"]
    neg_p, neu_p, pos_p = probs[0], probs[1], probs[2]

    # Clean Callout Status Banners
    if label == "positive":
        st.success("### Predicted Sentiment Response: ✨ POSITIVE")
    elif label == "neutral":
        st.info("### Predicted Sentiment Response: 🟡 NEUTRAL")
    else:
        st.error("### Predicted Sentiment Response: ❌ NEGATIVE")

    st.markdown(f"**Cleaned Sentence Analyzed:** *\"{res['cleaned']}\"*")
    
    # Highly Visible Metric Display Cards
    st.markdown("#### Model Class Confidence Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Positive Confidence Score", f"{pos_p * 100:.1f}%")
    m_col2.metric("Neutral Confidence Score", f"{neu_p * 100:.1f}%")
    m_col3.metric("Negative Confidence Score", f"{neg_p * 100:.1f}%")

    # Native Multi-Class Progress Bars
    st.progress(pos_p, text=f"Positive Match Probability: {pos_p * 100:.1f}%")
    st.progress(neu_p, text=f"Neutral Match Probability: {neu_p * 100:.1f}%")
    st.progress(neg_p, text=f"Negative Match Probability: {neg_p * 100:.1f}%")

    # Architecture Collapsible Diagnostics Panel
    with st.expander("🛠️ Active Model Structural Metadata", expanded=False):
        st.json({
            "Active Track Pipeline": res["model_label"],
            "Target Device Back-end": res["device"],
            "Hidden State Dim Size ($HIDDEN\_SIZE$)": res["cfg"]["hidden_size"],
            "Encoder Layers ($NUM\_LAYERS$)": res["cfg"]["num_layers"],
            "Attention Heads ($NUM\_HEADS$)": res["cfg"]["num_heads"],
            "Checkpoint Saved Validation Accuracy": f"{res['meta']['val_acc']:.4f}"
        })