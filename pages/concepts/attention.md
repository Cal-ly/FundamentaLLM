# Attention Mechanism

The attention mechanism is the core innovation that makes transformers powerful. Let's understand it from first principles.

## The Fundamental Problem

Traditional neural networks process each input independently. But in language, context matters:

```
"The bank was steep and muddy" 
vs
"I deposited money at the bank"
```

The word "bank" means different things based on context. How do we let the model understand this?

## The Solution: Attention

**Attention** lets each word "look at" and "focus on" other words in the sequence to understand its meaning in context.

### Intuition

Think of attention as asking questions:

```
For the word "bank":
- What is around me? → ["deposited", "money", "at", "the"]
- Which words are most relevant? → "deposited" (80%), "money" (15%), others (5%)
- Given this context, what should I represent? → financial institution
```

## How Attention Works

### Step 1: Create Query, Key, and Value

For each word, create three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

```
Word: "bank"
Query:  [0.2, 0.8, 0.1, ...]  ← What context do I need?
Key:    [0.5, 0.3, 0.9, ...]  ← What context do I provide?
Value:  [0.1, 0.7, 0.4, ...]  ← My actual information
```

**Why three vectors?** Separation of concerns:
- Query: What you're searching for
- Key: What others are advertising
- Value: The actual content to retrieve

### Step 2: Compute Attention Scores

Compare the Query of one word with the Keys of all words:

$$\text{score}(q, k) = q \cdot k$$

This dot product measures similarity: higher score = more relevant.

```
"bank" Query · "deposited" Key = 0.85  (high - relevant!)
"bank" Query · "the" Key       = 0.12  (low - less relevant)
"bank" Query · "money" Key     = 0.73  (high - relevant!)
```

### Step 3: Normalize with Softmax

Convert scores to probabilities (sum to 1):

$$\text{attention\_weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

The $\sqrt{d_k}$ scaling prevents scores from getting too large (which would make gradients vanish).

```
Before softmax: [0.85, 0.12, 0.73, ...]
After softmax:  [0.42, 0.03, 0.35, ...]  ← sums to 1.0
```

### Step 4: Weighted Sum of Values

Use attention weights to combine Values:

$$\text{output} = \sum_i \text{attention\_weight}_i \times \text{value}_i$$

```
"bank" representation = 
    0.42 × "deposited" Value +
    0.03 × "the" Value +
    0.35 × "money" Value +
    ...
```

Result: "bank" now has a representation influenced by "deposited" and "money" → financial context!

## Self-Attention: Complete Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Queries (what each token is looking for)
- $K$ = Keys (what each token advertises)
- $V$ = Values (actual information to retrieve)
- $d_k$ = dimension of keys (for scaling)

## Multi-Head Attention

Instead of one attention mechanism, use multiple in parallel (**heads**).

### Why Multiple Heads?

Different heads can focus on different aspects:

```
Head 1: Grammatical relationships
  "bank" attends to "the" (determiner)

Head 2: Semantic relationships  
  "bank" attends to "deposited", "money"

Head 3: Positional relationships
  "bank" attends to nearby words

Head 4: Syntactic relationships
  "bank" attends to subject/verb relationships
```

### Multi-Head Formula

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Process:**
1. Each head has its own learned weight matrices: $W_i^Q, W_i^K, W_i^V$
2. Each head computes attention independently
3. Concatenate all head outputs
4. Project with $W^O$ to final dimension

## Causal (Masked) Attention

For language modeling, we can't look at future tokens:

```
Position:     0    1    2    3
Tokens:       The  cat  sat  on
Allowed to
  see:        The  ←─┐
                   cat ←─┤
                      sat ←─┤
                         on ←─┘
```

**Implementation:** Set attention scores to $-\infty$ for future positions:

$$\text{scores}_{ij} = \begin{cases} 
q_i \cdot k_j & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}$$

After softmax, $-\infty$ becomes 0, so future tokens have no influence.

## Attention Visualization

### Example: "The quick brown fox jumped"

```
         The  quick brown  fox jumped
The      0.5   0.2   0.1   0.1   0.1     (focuses on itself mostly)
quick    0.2   0.4   0.3   0.1   0.0     (focuses on "brown" too)
brown    0.1   0.2   0.4   0.3   0.0     (focuses on "fox")
fox      0.1   0.1   0.2   0.5   0.1     (focuses on itself)
jumped   0.1   0.1   0.1   0.2   0.5     (focuses on itself)
```

Numbers show attention weights: how much each word (row) attends to each word (column).

## Implementation in FundamentaLLM

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Create Q, K, V projections for all heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project and split into heads
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        output = output.view(batch_size, seq_len, d_model)
        return self.out_proj(output)
```

## Why Attention Works

### 1. **Parallel Processing**
All positions computed simultaneously (unlike RNNs which are sequential).

### 2. **Long-Range Dependencies**
Any token can directly attend to any other token (no information loss over distance).

### 3. **Flexible Context**
Attention weights are data-dependent (each input has custom attention pattern).

### 4. **Interpretable**
Attention weights show what the model is "looking at."

## Computational Complexity

**Time:** $O(n^2 \cdot d)$ where $n$ = sequence length, $d$ = model dimension

**Space:** $O(n^2)$ for storing attention matrix

This is why long sequences are expensive: attention grows quadratically with length.

## Attention Variants

### Standard (Bidirectional) Attention
- Each token can see all tokens
- Used in: BERT, encoders

### Causal (Autoregressive) Attention
- Each token can only see previous tokens
- Used in: GPT, decoders
- **FundamentaLLM uses this**

### Cross Attention
- Attend from one sequence to another
- Used in: Encoder-decoder models

## Common Pitfalls

### 1. Forgetting Scaling
Without $\sqrt{d_k}$ scaling, softmax saturates and gradients vanish.

### 2. Wrong Mask Shape
Causal mask must be triangular, not just blocking future tokens.

### 3. Dimension Mismatches
`num_heads` must divide `d_model` evenly.

## Key Insights

1. **Attention is learned lookup**: Query searches Keys to retrieve Values
2. **Multi-head = multiple perspectives**: Each head learns different patterns
3. **Softmax normalizes**: Attention is a probability distribution
4. **Masking enforces causality**: Can't peek at future tokens

## Comparing to Other Mechanisms

| Mechanism | How it connects tokens | Complexity |
|-----------|----------------------|------------|
| RNN | Sequential, state-based | $O(n)$ but sequential |
| CNN | Local windows | $O(n)$ but limited range |
| Attention | All-to-all, direct | $O(n^2)$ but parallel |

Attention trades computation (quadratic) for capability (any-to-any connections).

## Further Reading

- "Attention is All You Need" (Vaswani et al., 2017) - Original transformer paper
- [Illustrated Attention](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Next Steps

- [Positional Encoding](./positional-encoding.md) - How transformers encode position
- [Transformer Architecture](./transformers.md) - How attention fits into the full model
- [Models Module](../modules/models.md) - Implementation details
