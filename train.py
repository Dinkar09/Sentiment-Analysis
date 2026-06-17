import math
import torch
import torch.nn as nn

# 1. Embeddings Module

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, type_vocab_size, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.seg_embed = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, segment_ids):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Combine token, positional, and segment embeddings
        tok_emb = self.tok_embed(x)
        pos_emb = self.pos_embed(pos)
        seg_emb = self.seg_embed(segment_ids)
        
        x = tok_emb + pos_emb + seg_emb
        x = self.layer_norm(x)
        return self.dropout(x)

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Project linear and split into heads
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v)
        
        # Concat and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.out_linear(context)

# 3. Feed Forward Network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))

# 4. Single Transformer Block (Encoder Layer)
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# 5. Complete BERT Module
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_len, type_vocab_size, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_len, type_vocab_size, dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout) 
            for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
        
    def forward(self, input_ids, segment_ids, mask=None):
        """
        BERT Forward Pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            segment_ids: Segment labels [batch_size, seq_len]
            mask: Attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Tuple: (sequence_output, pooled_output)
        """
        
        # STEP 1: INPUT EMBEDDINGS
        x = self.embeddings(input_ids, segment_ids)
        
        # STEP 2: ENCODER STACK (12 Transformer Blocks)
        for layer in self.layers:
            x = layer(x, mask)
        
        # STEP 3: SEQUENCE OUTPUT
        sequence_output = x
        
        # STEP 4: POOLED OUTPUT (For Classification)
        first_token_tensor = x[:, 0]
        pooled_output = self.pooler_activation(self.pooler(first_token_tensor))
        
        return sequence_output, pooled_output

# ==========================================
# Training Entry Point
# ==========================================
if __name__ == "__main__":
    from tracing import BertExecutionTracker
    
    # Hyperparameters for BERT-Base
    VOCAB_SIZE = 30000
    HIDDEN_SIZE = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12
    INTERMEDIATE_SIZE = 3072
    MAX_LEN = 512
    TYPE_VOCAB_SIZE = 2
    
    # Create Dummy Input
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 128))       # 2 sequences, length 128
    segment_ids = torch.randint(0, TYPE_VOCAB_SIZE, (2, 128))
    attention_mask = torch.ones(2, 128).unsqueeze(1).unsqueeze(2) # (Batch, 1, 1, Seq_len)
    
    # Instantiate the model
    bert_model = BERT(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, INTERMEDIATE_SIZE, MAX_LEN, TYPE_VOCAB_SIZE)
    
    # Initialize tracker and run BERT with execution tracing
    tracker = BertExecutionTracker(output_dir="bert_execution_outputs")
    sequence_output, pooled_output = tracker.trace_forward_pass(bert_model, input_ids, segment_ids, attention_mask)
    
    # Display final results
    print("FINAL RESULTS:")
    print(f"  Sequence Output Shape: {sequence_output.shape} # [Batch, Seq_Len, Hidden_Size]")
    print(f"  Pooled Output Shape: {pooled_output.shape}     # [Batch, Hidden_Size]")