# Language Modeling

Language modeling is the core task that powers large language models. Understanding it is key to understanding how LLMs learn.

## What is Language Modeling?

**Objective:** Given a sequence of tokens, predict the next token.

```
Input:  "The cat sat on the"
Task:   Predict next word
Output: "mat" (or "chair", "floor", etc.)
```

That's it. This simple task, when scaled up with billions of parameters and trillions of tokens, creates models like GPT.

## Why Does This Work?

To predict the next word well, the model must:

1. **Understand grammar** - "the" is followed by a noun
2. **Track context** - We're talking about a cat sitting
3. **Learn semantics** - Cats sit on physical surfaces
4. **Model world knowledge** - Common places cats sit

By learning to predict the next token, the model learns language structure, meaning, and even facts about the world.

## Formal Definition

Given a sequence of tokens $x_1, x_2, ..., x_{n-1}$, predict the probability distribution over the next token $x_n$:

$$P(x_n | x_1, x_2, ..., x_{n-1})$$

The model outputs a probability for every possible token:

```
P("mat" | "The cat sat on the") = 0.35
P("floor" | "The cat sat on the") = 0.25
P("chair" | "The cat sat on the") = 0.20
P("moon" | "The cat sat on the") = 0.0001  ← unlikely!
...
```

## Autoregressive Generation

Language models generate text **autoregressively** - one token at a time, using previous predictions as input:

```
1. Start: "The"
2. Predict next: "cat"     → "The cat"
3. Predict next: "sat"     → "The cat sat"
4. Predict next: "on"      → "The cat sat on"
5. Predict next: "the"     → "The cat sat on the"
6. Predict next: "mat"     → "The cat sat on the mat"
```

Each prediction is conditioned on all previous tokens.

## Training Objective

### Cross-Entropy Loss

The standard loss for language modeling is **cross-entropy** between predicted and actual next token:

$$\mathcal{L} = -\log P(x_n | x_1, ..., x_{n-1})$$

**Intuition:** Minimize the negative log-likelihood of the correct token.

If model predicts:
- Correct token with high probability → Low loss (good!)
- Correct token with low probability → High loss (bad!)

### Example

```
True next token: "mat" (ID: 1523)

Model predictions:
  Token 1523 ("mat"):   0.35  → Loss = -log(0.35) = 1.05
  Token 891 ("floor"):  0.25
  Token 234 ("chair"):  0.20
  ...

Better predictions:
  Token 1523 ("mat"):   0.85  → Loss = -log(0.85) = 0.16 ✓
```

Lower loss = better prediction.

## Perplexity

**Perplexity** is the exponentiated loss, measuring how "surprised" the model is:

$$\text{Perplexity} = \exp(\mathcal{L})$$

**Interpretation:**
- Perplexity of 1: Model is certain (perfect prediction)
- Perplexity of 100: Model is choosing among ~100 likely options
- Lower perplexity = better model

**Typical values:**
- Random guessing (256 chars): Perplexity = 256
- Decent character model: Perplexity = 3-10
- Good character model: Perplexity = 1.5-3

## Training Process

### 1. Create Input-Target Pairs

From text "The cat sat":

```
Input:  "T"        → Target: "h"
Input:  "Th"       → Target: "e"
Input:  "The"      → Target: " "
Input:  "The "     → Target: "c"
Input:  "The c"    → Target: "a"
Input:  "The ca"   → Target: "t"
Input:  "The cat"  → Target: " "
Input:  "The cat " → Target: "s"
...
```

Every position in the text creates a training example.

### 2. Batching with Masking

Process multiple sequences in parallel:

```
Batch of sequences (max length 5):
┌─────────────────────────────────┐
│ T  h  e     c  →  Predict: a    │
│ T  h  i  s     →  Predict: (space) │
│ H  e  l  l  o  →  Predict: !    │
└─────────────────────────────────┘

Causal masking ensures:
- Position 0 only sees position 0
- Position 1 sees positions 0-1
- Position 2 sees positions 0-2
- etc.
```

### 3. Forward Pass

```
Input tokens → Embed → Transformer → Output logits

Logits shape: [batch_size, seq_len, vocab_size]
Each position predicts distribution over all tokens
```

### 4. Compute Loss

```python
# Get predictions for all positions
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# Shift to align input and target
# Input:  [T, h, e,  ]  →  Logits for: [h, e,  , c]
# Target: [h, e,  , c]
predictions = logits[:, :-1, :]
targets = input_ids[:, 1:]

# Cross-entropy loss
loss = F.cross_entropy(
    predictions.reshape(-1, vocab_size),
    targets.reshape(-1)
)
```

### 5. Backpropagation

Gradients flow through:
- Output layer
- Transformer layers (attention + FFN)
- Embeddings

Model weights updated to increase probability of correct next tokens.

## Why Character-Level Modeling?

FundamentaLLM uses character-level language modeling:

```
Input:  "H e l l"
Target: "e l l o"
```

**Advantages:**
- Simple and universal
- No vocabulary management
- Handles any text

**Challenges:**
- Longer sequences (more characters than words)
- Must learn character→word→syntax hierarchy
- More training needed to capture long-range dependencies

## Evaluation Metrics

### 1. Perplexity
How uncertain the model is (lower = better).

### 2. Bits Per Character (BPC)
Related to perplexity:
$$\text{BPC} = \log_2(\text{perplexity})$$

Typical values: 1.0-2.0 bits/char for good character models.

### 3. Generation Quality
Does generated text look reasonable?

```
Good model: "The cat sat on the mat and purred softly."
Bad model:  "The cat x2@ q1 zrp mat potato."
```

## Conditional Language Modeling

Can condition on additional context:

### Prompted Generation
```
Prompt: "Write a story about a robot:"
Continue: "Once upon a time, a robot named Bob..."
```

### Instruction Following
```
Instruction: "Translate to French:"
Input: "Hello"
Output: "Bonjour"
```

Same basic objective (predict next token), but with structured conditioning.

## Fine-Tuning vs Pre-Training

### Pre-Training
Train on massive amounts of text to learn general language patterns:
```
Data: Books, Wikipedia, web pages, code
Task: Predict next token
Result: General language understanding
```

### Fine-Tuning
Train on specific task data:
```
Data: Question-answer pairs
Task: Still predict next token, but on Q&A format
Result: Specialized behavior
```

**FundamentaLLM focuses on pre-training** - learning language from scratch.

## Common Patterns the Model Learns

Through next-token prediction, models learn:

### 1. Syntax
```
"The cat is" → "sleeping" (verb form matches)
Not: "The cat is sleeps" ✗
```

### 2. Semantics
```
"Fire is" → "hot" (semantic knowledge)
Unlikely: "Fire is cold" ✗
```

### 3. Pragmatics
```
Q: "What's your name?"
A: "My name is..." (appropriate response format)
```

### 4. World Knowledge
```
"The capital of France is" → "Paris"
"Einstein was born in" → "1879"
```

All from predicting the next token!

## Why This is Powerful

Language modeling as pre-training:

1. **Unlabeled data** - Don't need human annotations, just text
2. **Scalable** - Can train on trillions of tokens
3. **Transfer learning** - Pre-trained models adapt to specific tasks
4. **Emergent abilities** - Complex behaviors emerge from simple objective

## Implementation Example

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = Transformer(d_model, num_layers=6)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Process through transformer
        x = self.transformer(x)
        
        # Predict next token logits
        logits = self.output(x)
        
        return logits
    
    def compute_loss(self, input_ids):
        # Get predictions
        logits = self.forward(input_ids)
        
        # Shift targets (predict next token)
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        return loss
```

## Key Insights

1. **Simple objective, complex learning** - Next token prediction leads to language understanding
2. **Autoregressive = sequential** - Generate one token at a time
3. **Cross-entropy optimizes predictions** - Maximize probability of correct token
4. **Perplexity measures uncertainty** - Lower is better
5. **Scale matters** - More data + bigger models = better language modeling

## Comparison to Other Tasks

| Task | Objective | Use Case |
|------|-----------|----------|
| Language Modeling | Predict next token | General language understanding |
| Classification | Predict category | Sentiment, topics, etc. |
| Translation | Seq-to-seq mapping | Convert languages |
| Question Answering | Extract/generate answer | Specific info retrieval |

Language modeling is the foundation - other tasks can be framed as special cases.

## Limitations

1. **Compute intensive** - O(n²) attention for sequence length n
2. **No explicit reasoning** - Just pattern matching
3. **Training data biases** - Learns whatever is in the data
4. **No grounding** - Doesn't "understand" meaning (debatable!)

## Further Reading

- "Attention is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3 paper)
- [Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Next Steps

- [Autoregressive Generation](./autoregressive.md) - How to generate text
- [Loss Functions](./losses.md) - Deep dive into training objectives
- [Training Module](../modules/training.md) - Implementation details
