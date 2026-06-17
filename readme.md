# User Guide: BERT Sentiment Analysis Hub
## Project Structure, Workflow, and Technical Architecture Overview

Welcome to the comprehensive Project Architecture Blueprint for your **BERT Sentiment Analysis Hub**. I have prepared this document to give you a transparent, clear, and high-level structural breakdown of the codebase. 

This system represents an enterprise-grade, end-to-end Machine Learning pipeline. It cleans raw natural language data, transforms it into deep mathematical vectors using advanced BERT (Bidirectional Encoder Representations from Transformers) tokenization, executes dual-track model training optimized for specific hardware footprints, and delivers a polished web interface for cross-model verification.

---

## 1. High-Level System Architecture

Before diving into individual files, it is vital to understand how information flows through the system. Below is an architectural overview mapping out how your text travels from a raw customer review into an automated sentiment classification matrix.

```
 [ Raw Data Input ] ──> sentiment_data.csv (Customer sentences & feedback)
                                │
                                ▼
 [ Processing Layer ] ──> preprocessing.py (Removes noise, normalizes text)
                                │
                                ▼
                       Preprocessed data/bert_tokenised_output.json (Clean tensors)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
 [ Model Track A ]                           [ Model Track B ]
   cpuTrained.py                               gpuTrained.py
   (Resource-light/Localized)                  (Hardware-accelerated/Deeper)
          │                                           │
          ▼                                           ▼
   checkpoints/best_model.pt                   checkpoints/best_model_gpu.pt
          │                                           │
          └─────────────────────┬─────────────────────┘
                                │ (Dynamic Run-Time Injection)
                                ▼
 [ Interface Layer ] ───> ui.py (Vertical Streamlit Dashboard Engine)
                                │
                                ▼
                     [ Client Desktop/Web ]
```

---

## 2. Directory & Blueprint Topography

Here is the exact layout of the codebase. Each file is decoupled from the others to allow engineers to upgrade the text processing rulebook, alter neural network hyperparameters, or re-skin the visual interface without breaking other components in the system.

```
sentiment-analysis-hub/
├── Preprocessed data/
│   └── bert_tokenised_output.json      # Output from the tokenization matrix
├── bert_execution_outputs/
│   └── BERT_FINAL_OUTPUT.json          # System internal tracing & validation matrix
├── checkpoints/
│   ├── best_model.pt                   # Model 1 weights (CPU Optimized, 4 layers)
│   └── best_model_gpu.pt               # Model 2 weights (GPU Accelerated, 6 layers)
├── cpuTrained.py                       # Lightweight training pipeline script
├── gpuTrained.py                       # High-performance training pipeline script
├── preprocessing.py                    # Structured cleaning & validation script
├── sentiment_data.csv                  # Core baseline labeled source data
├── tracing.py                          # Diagnostic interceptor for layer weights
└── ui.py                               # Production Streamlit client interface
```

---

## 3. Core Component Deep Dive

### 📊 Baseline Data Layer (`sentiment_data.csv`)
* **Purpose:** This serves as the single source of truth for training the system.
* **Content:** It houses thousands of rows of real-world sentences mapped against deterministic target classes.
* **Label Strategy:**
  * `0` $
ightarrow$ **Negative Sentiment** (e.g., product failure logs, complaints)
  * `1` $
ightarrow$ **Neutral Sentiment** (e.g., standard factual inquiries, observations)
  * `2` $
ightarrow$ **Positive Sentiment** (e.g., high-satisfaction user milestones)

### 🧹 Text Normalization Engine (`preprocessing.py`)
* **Purpose:** Raw human input is full of noise (URLs, tags, random spacing). This script standardizes text into matrices that can be processed by neural networks.
* **The 7-Step Cleaning Rig:**
  1. **Case Unification:** Converts all sentences to lower-case.
  2. **URL Erasure:** Strips web domains (`http`, `https`, `www`) because links do not carry emotional intent.
  3. **HTML Stripping:** Removes markup code tags (such as `<br>`, `&amp;`) often found in web scraping datasets.
  4. **Social Handle Removal:** Cleans out `@mentions` to keep personal names from biasing text scores.
  5. **Hashtag Preservation:** Extracts structural keywords by stripping the `#` symbol but preserving the semantic string behind it.
  6. **Character Filtering:** Filters out emojis and strange special symbols, leaving behind standard English syntax and basic punctuation.
  7. **White-Space Compression:** Compresses multi-space gaps down to simple single-character gaps.
* **Tokenization:** Converts the cleaned strings into standard `bert-base-uncased` integers (`input_ids`, `attention_mask`) padded to an absolute block length of `128` items, and saves them to the `Preprocessed data/` directory.

### 🧠 Track 1: Localized Computations (`cpuTrained.py`)
* **Purpose:** Built to run on basic server infrastructure or local environments without needing expensive hardware.
* **Architecture Design:** Configured as a highly responsive, custom, lightweight BERT model consisting of **4 Transformer Blocks** and **4 Self-Attention Heads** with a hidden vector space layer of `256`.
* **Execution Behavior:** Splits your data into cross-validation sets, trains over 5 epochs using an AdamW optimizer, checks validation accuracy iteratively, and outputs optimal performance weights into `checkpoints/best_model.pt`.

### ⚡ Track 2: High-Performance Acceleration (`gpuTrained.py`)
* **Purpose:** Built to leverage modern hardware accelerators (NVIDIA CUDA VRAM architectures) for fast scale-up.
* **Architecture Design:** Built as a larger model containing **6 Deep Transformer Blocks** and **6 Self-Attention Heads** with an expanded hidden layer size of `384`.
* **Hardware Optimizations:**
  * **AMP (Automatic Mixed Precision):** Dynamically changes standard 32-bit floats down to 16-bit math operations where possible, cutting active VRAM consumption in half and doubling training speeds.
  * **Memory Pinning:** Locks host RAM directly to prevent system copying delays between host processors and physical graphic processing units.
  * **cuDNN Benchmarking:** Flags the system runtime to scan hardware characteristics to deploy optimal matrix manipulation algorithms.
* **Output:** Saves high-fidelity neural network states into `checkpoints/best_model_gpu.pt`.

### 🔍 System Inspector (`tracing.py`)
* **Purpose:** A developer tool that lets engineers peek inside the model's layers.
* **Behavior:** Intercepts the forward execution stream of the neural encoder during runtime. It extracts the raw mathematical representations generated for the primary classification token (`[CLS]`) and outputs a diagnostic snapshot to `BERT_FINAL_OUTPUT.json`.

### 🖥️ Customer Evaluation Dashboard (`ui.py`)
* **Purpose:** The actual interface for your users. It connects your technical model back-end directly to an intuitive, non-technical dashboard.
* **How It Works (Strict Vertical Hierarchy):**
  1. **Dynamic Architecture Scanner:** When launched, the interface doesn't hardcode model shapes. It inspects the `.pt` files to identify layer settings automatically.
  2. **Model Selection Panel:** Users toggle between `CPU trained model` or `GPU trained model`.
  3. **Sentence Input Box:** A clean text entry zone with intuitive placeholders.
  4. **Analysis Core:** Clicking the primary execution trigger cleans the input string on-the-fly and runs a forward mathematical pass.
  5. **Clean Native Metrics Reporting:** Renders visual status callouts alongside confidence percentage metrics, preventing raw code leaks while providing clear performance visibility.

---

## 4. Operational Workflow for Engineering Teams

To execute, initialize, or retrain the system platform, developers run the core processing scripts in this exact sequence:

```
[ Step 1: Data Preparation ]
  $ python preprocessing.py
  ├── Reads: sentiment_data.csv
  └── Creates: Preprocessed data/bert_tokenised_output.json

[ Step 2: Training & Optimization ]
  $ python cpuTrained.py   ---> Generates checkpoints/best_model.pt
  $ python gpuTrained.py   ---> Generates checkpoints/best_model_gpu.pt

[ Step 3: Production UI Deployment ]
  $ streamlit run ui.py
  ├── Dynamically reads structural parameters from checkpoints
  └── Spins up the web application browser interface
```

---

## 5. Strategic Benefits for Your Business

1. **Strategic Cost Control:** By decoupling the codebase into distinct CPU and GPU workflows, your DevOps team can run cost-effective local microservices for baseline analysis, spinning up more resource-intensive GPU operations only when processing high-volume enterprise queues.
2. **Modular Architecture:** The preprocessing engine is fully separated from the training modules. If your business scales into international markets requiring language adjustments, developers can easily swap out the un-cased tokenizer layer without rewriting the core dashboard code.
3. **Hardware Agnostic Runtime Configuration:** The user interface relies on dynamic asset loading. As your engineers refine newer versions of the neural architecture, you can drop updated weights into the checkpoints folder, and the application will instantly adapt to the new layout without requiring code rewrites.

---

---

## 6. Parameters learned by each Model

**Both your CPU-trained BERT-Small (256 hidden, 4 layers, 4 heads) and GPU-trained BERT-Small (384 hidden, 6 layers, 6 heads) learned 10,022,403 and approximately 34,869,251 trainable parameters respectively.**

---