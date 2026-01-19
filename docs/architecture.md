# Architecture Guide

## Overview

FundamentaLLM implements a decoder-only transformer architecture similar to GPT models. This guide explains the design decisions, component implementations, and data flow through the system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FundamentaLLM Pipeline                   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Data Processing                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Raw Text   │→ │  Tokenizer   │→ │   Dataset    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Model Components                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Embeddings  │→ │ Transformer  │→ │  LM Head     │     │
│  │  + Positional│  │   Blocks     │  │  (Linear)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Training                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Loss       │→ │  Optimizer   │→ │  Scheduler   │     │
│  │ (Cross Ent.) │  │   (AdamW)    │  │  (Cosine)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Generation                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Sampling   │→ │  Generator   │→ │   Output     │     │
│  │  (Top-k/p)   │  │              │  │    Text      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Tokenization

**File:** `src/fundamentallm/data/tokenizers/character.py`

FundamentaLLM uses character-level tokenization for simplicity and educational clarity.

**Design:**
```python
class CharacterTokenizer:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(vocab)}
```

**Process:**
1. **Training:** Extract unique characters from corpus → build vocabulary
2. **Encoding:** Map each character to integer ID
3. **Decoding:** Map integer IDs back to characters

**Example:**
```python
text = "Hello"
tokenizer.train([text])
# vocab: ['H', 'e', 'l', 'o', '<pad>', '<unk>']

encoded = tokenizer.encode(text)
# [0, 1, 2, 2, 3]  # H=0, e=1, l=2, o=3

decoded = tokenizer.decode(encoded)
# "Hello"
```

**Special Tokens:**
- `<pad>`: Padding for batching (ID=vocab_size-2)
- `<unk>`: Unknown characters (ID=vocab_size-1)

**Why Character-Level?**
- ✅ No out-of-vocabulary issues
- ✅ Handles any text without preprocessing
- ✅ Simple to understand and implement
- ❌ Longer sequences than word/subword tokens
- ❌ Must learn character→word relationships

---

### 2. Data Pipeline

**File:** `src/fundamentallm/data/dataset.py`

**TextDataset:**
```python
class TextDataset(Dataset):
    def __init__(self, tokens: List[int], sequence_length: int):
        self.tokens = tokens
        self.sequence_length = sequence_length
    
    def __getitem__(self, idx: int):
        start = idx
        end = start + self.sequence_length
        
        x = self.tokens[start:end]      # Input
        y = self.tokens[start+1:end+1]  # Target (shifted by 1)
        
        return torch.tensor(x), torch.tensor(y)
```

**Key Concepts:**

**Sequence Length:** Fixed-length windows for training
```
Text:     "Hello world!"
Tokens:   [5, 8, 11, 11, 14, 22, 14, 17, 11, 3, 25]

Sequence length = 4:
Sample 0: x=[5, 8, 11, 11],    y=[8, 11, 11, 14]
Sample 1: x=[8, 11, 11, 14],   y=[11, 11, 14, 22]
Sample 2: x=[11, 11, 14, 22],  y=[11, 14, 22, 14]
...
```

**DataLoader:** Batching + shuffling
```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,    # Random order each epoch
    num_workers=4,   # Parallel loading
)

# Batch shape: (batch_size, sequence_length)
for x, y in loader:
    # x.shape: (32, 128)
    # y.shape: (32, 128)
    pass
```

---

### 3. Transformer Model

**File:** `src/fundamentallm/models/transformer.py`

#### 3.1 Token + Positional Embeddings

**File:** `src/fundamentallm/models/components/embeddings.py`

```python
class TokenEmbeddings(nn.Module):
    def forward(self, x):
        # x: (batch, seq_len) - integer token IDs
        # output: (batch, seq_len, d_model) - continuous vectors
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def forward(self, x):
        # Add sinusoidal position encodings
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        return x + self.pe[:, :x.size(1), :]
```

**Why Positional Encoding?**
- Transformer has no inherent notion of order (unlike RNNs)
- Sine/cosine patterns encode position information
- Allows model to use position in attention

**Visualization:**
```
Token IDs:        [5,    8,    11,   14]
                   ↓     ↓     ↓     ↓
Token Embeddings: [e₅,  e₈,   e₁₁,  e₁₄]  (d_model dims each)
                   +     +     +     +
Pos Encodings:    [p₀,  p₁,   p₂,   p₃]   (sinusoidal)
                   =     =     =     =
Final:            [x₀,  x₁,   x₂,   x₃]   → Transformer Blocks
```

#### 3.2 Multi-Head Attention

**File:** `src/fundamentallm/models/components/attention.py`

**Core Idea:** Learn to focus on relevant parts of the sequence.

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x, mask=None):
        # 1. Linear projections: Q, K, V
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Split into multiple heads
        Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
        K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
        V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq, d_k)
        
        # 3. Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)
        # Shape: (batch, num_heads, seq, seq)
        
        # 4. Apply causal mask (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax + weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V
        # Shape: (batch, num_heads, seq, d_k)
        
        # 6. Concatenate heads + final projection
        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch, seq, d_model)
        return self.out_proj(output)
```

**Causal Masking:**
```
Position:  0   1   2   3
           ┌───┬───┬───┬───┐
        0  │ ✓ │ ✗ │ ✗ │ ✗ │  Token 0 can only attend to itself
           ├───┼───┼───┼───┤
        1  │ ✓ │ ✓ │ ✗ │ ✗ │  Token 1 can attend to 0, 1
           ├───┼───┼───┼───┤
        2  │ ✓ │ ✓ │ ✓ │ ✗ │  Token 2 can attend to 0, 1, 2
           ├───┼───┼───┼───┤
        3  │ ✓ │ ✓ │ ✓ │ ✓ │  Token 3 can attend to all previous
           └───┴───┴───┴───┘
```

Prevents information leakage from future tokens during training.

**Multi-Head Intuition:**
- Each head learns different relationships (syntax, semantics, etc.)
- Parallel processing increases model capacity
- Default: 8 heads with d_k = d_model / num_heads

#### 3.3 Feed-Forward Network

**File:** `src/fundamentallm/models/components/feedforward.py`

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Expand to d_ff dimensions
        x = F.gelu(self.linear1(x))  # (batch, seq, d_ff)
        x = self.dropout(x)
        # Project back to d_model
        x = self.linear2(x)          # (batch, seq, d_model)
        return self.dropout(x)
```

**Why FFN?**
- Applies non-linear transformation to each position independently
- Increases model expressiveness
- Typical expansion: d_ff = 4 × d_model

#### 3.4 Transformer Block

**File:** `src/fundamentallm/models/transformer.py`

```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        # 1. Multi-head attention with residual + layer norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)  # Residual connection
        
        # 2. Feed-forward with residual + layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)    # Residual connection
        
        return x
```

**Layer Normalization:**
```python
# Normalize across d_model dimension for each token
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + eps)
```

**Residual Connections:**
- Enable gradient flow through deep networks
- Allow information to bypass transformations
- Critical for training stability

**Visualization:**
```
Input x
   │
   ├──────────────┐
   │              │
   │   Multi-Head │
   │   Attention  │
   │              │
   └──────(+)─────┘  Add
          │
      LayerNorm
          │
   ├──────────────┐
   │              │
   │ Feed-Forward │
   │              │
   └──────(+)─────┘  Add
          │
      LayerNorm
          │
      Output x'
```

#### 3.5 Complete Model

```python
class TransformerLM(nn.Module):
    def __init__(self, config):
        self.token_embeddings = TokenEmbeddings(...)
        self.pos_encoding = PositionalEncoding(...)
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # 1. Embed tokens + add positions
        x = self.token_embeddings(x)
        x = self.pos_encoding(x)
        
        # 2. Pass through transformer blocks
        mask = self.create_causal_mask(x.size(1))
        for block in self.blocks:
            x = block(x, mask)
        
        # 3. Final layer norm
        x = self.norm(x)
        
        # 4. Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq, vocab_size)
        
        return logits
```

---

### 4. Training

**File:** `src/fundamentallm/training/trainer.py`

#### 4.1 Loss Function

Cross-entropy loss for language modeling:

```python
def compute_loss(logits, targets):
    # logits: (batch, seq, vocab_size)
    # targets: (batch, seq)
    
    # Flatten for cross-entropy
    logits = logits.view(-1, vocab_size)   # (batch*seq, vocab_size)
    targets = targets.view(-1)             # (batch*seq,)
    
    loss = F.cross_entropy(logits, targets)
    return loss
```

**Interpretation:**
- Lower loss = better prediction of next token
- Perplexity = exp(loss) = interpretable metric
- Target: minimize average negative log-likelihood

#### 4.2 Optimizer

AdamW (Adam with weight decay):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,  # L2 regularization
)
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Handles sparse gradients well
- Decoupled weight decay for better regularization

#### 4.3 Learning Rate Schedule

Warmup + Cosine Annealing:

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

**Visualization:**
```
LR
│
│     ┌────────────────────────┐
│    ╱                          ╲
│   ╱                            ╲
│  ╱                              ╲
│ ╱                                ╲___________
└──────────────────────────────────────────────► Step
  ←warmup→   ←───── cosine decay ──────→
```

#### 4.4 Training Loop

```python
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    
    for x, y in loader:
        # Forward pass
        logits = model(x)
        loss = compute_loss(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

---

### 5. Generation

**File:** `src/fundamentallm/generation/generator.py`

#### 5.1 Autoregressive Generation

```python
def generate(model, prompt_ids, max_tokens):
    model.eval()
    generated = prompt_ids.clone()
    
    for _ in range(max_tokens):
        # 1. Get logits for last token
        logits = model(generated)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)
        
        # 2. Sample next token
        next_token = sample(next_token_logits)
        
        # 3. Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return generated
```

#### 5.2 Sampling Strategies

**Greedy (deterministic):**
```python
next_token = torch.argmax(logits)
```
- Always picks highest probability token
- Repetitive output
- Good for debugging

**Temperature Sampling:**
```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```
- `temperature < 1.0`: More confident (sharper distribution)
- `temperature > 1.0`: More random (flatter distribution)
- `temperature = 1.0`: Standard sampling

**Top-k Sampling:**
```python
top_k_logits, top_k_indices = torch.topk(logits, k=50)
probs = F.softmax(top_k_logits, dim=-1)
next_token_idx = torch.multinomial(probs, num_samples=1)
next_token = top_k_indices[next_token_idx]
```
- Only sample from top-k most likely tokens
- Filters out improbable tokens
- k=50 is common default

**Top-p (Nucleus) Sampling:**
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > p

# Remove tokens outside nucleus
sorted_logits[sorted_indices_to_remove] = float('-inf')
probs = F.softmax(sorted_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```
- Sample from smallest set of tokens with cumulative prob > p
- Adapts to distribution shape (unlike fixed top-k)
- p=0.95 is common default

**Recommended settings:**
```python
temperature = 0.8  # Slightly confident
top_k = 50        # Filter unlikely tokens
top_p = 0.95      # Nucleus sampling
```

---

## Design Decisions

### Why Character-Level Tokenization?

**Pros:**
- ✅ No vocabulary size limit
- ✅ Handles any Unicode text
- ✅ No preprocessing needed
- ✅ Educational simplicity

**Cons:**
- ❌ Longer sequences (more memory)
- ❌ Must learn character→word mappings
- ❌ Slower than subword tokenization

**Alternative:** Byte-pair encoding (BPE) or SentencePiece for production.

### Why Decoder-Only Architecture?

**Decoder-only (GPT-style):**
- Causal (can only see past)
- Good for generation
- Simpler architecture

**Encoder-decoder (BERT-style):**
- Bidirectional attention
- Good for understanding tasks
- More complex

**Choice:** Decoder-only for educational focus on text generation.

### Why Sinusoidal Positional Encoding?

**Sinusoidal (used here):**
- ✅ No learned parameters
- ✅ Can extrapolate to longer sequences
- ✅ Mathematical elegance

**Learned positional embeddings (alternative):**
- ✅ Slightly better performance
- ❌ Fixed maximum sequence length
- ❌ Extra parameters to train

**Choice:** Sinusoidal for simplicity and extrapolation ability.

### Why Layer Normalization?

**Layer Norm (used here):**
- Normalizes across feature dimension
- Works well with variable batch sizes
- Standard for transformers

**Batch Norm (alternative):**
- Normalizes across batch dimension
- Doesn't work well with small batches
- Better for CNNs

**Choice:** Layer norm for transformer stability.

---

## Mathematical Formulation

### Attention Mechanism

**Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$: Query matrix (batch, seq, d_k)
- $K$: Key matrix (batch, seq, d_k)
- $V$: Value matrix (batch, seq, d_k)
- $d_k$: Dimension per head

**Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Feed-Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with GELU activation:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

### Positional Encoding

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu$: Mean across feature dimension
- $\sigma^2$: Variance across feature dimension
- $\gamma, \beta$: Learned scale and shift

---

## Parameter Count

For a model with:
- vocab_size = 256
- d_model = 512
- num_heads = 8
- num_layers = 6
- d_ff = 2048

**Breakdown:**
```
Token embeddings:     256 × 512 = 131K
Positional encoding:  0 (no params)

Per transformer block:
  - Attention QKV:    3 × (512 × 512) = 786K
  - Attention out:    512 × 512 = 262K
  - FFN 1:            512 × 2048 = 1,049K
  - FFN 2:            2048 × 512 = 1,049K
  - LayerNorms:       4 × 512 = 2K
  Block total:        ~3,148K

6 blocks:             6 × 3,148K = 18,888K
LM head:              512 × 256 = 131K

Total:                ~19,150K ≈ 19M parameters
```

**Memory usage:**
- Float32: 19M × 4 bytes = 76 MB
- Float16: 19M × 2 bytes = 38 MB

---

## Performance Considerations

### Attention Complexity

**Time complexity:** $O(n^2 \cdot d)$
- $n$: Sequence length
- $d$: Model dimension

**Problem:** Quadratic in sequence length

**Solutions:**
- Limit sequence length (e.g., 512 tokens)
- Use efficient attention variants (future work)
- Batch processing for parallelism

### Memory Bottlenecks

**Largest memory consumers:**
1. **Attention scores:** (batch, heads, seq, seq)
2. **Activations:** Stored for backprop
3. **Optimizer states:** 2× parameters for Adam

**Optimization strategies:**
- Mixed precision (FP16)
- Gradient accumulation
- Gradient checkpointing (future)

---

## Next Steps

- **[Training Guide](training_guide.md)** - Best practices for training
- **[Getting Started](getting_started.md)** - Quick start tutorial
- **[Example Notebooks](notebooks/)** - Interactive walkthroughs

---

**References:**
- Vaswani et al. (2017). "Attention Is All You Need"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners"
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
