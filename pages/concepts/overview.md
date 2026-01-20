# Concepts Overview

This section explains the theoretical foundations of FundamentaLLM. Whether you're new to transformers or exploring implementation details, start here.

## Learning Path

### Beginner: "What's a Transformer?"

1. **[Transformer Architecture](./transformers.md)** 
   - What is a transformer?
   - Why is it useful?
   - How do the pieces fit together?

2. **[Attention Mechanism](./attention.md)**
   - The core innovation of transformers
   - How attention "focuses" on relevant parts
   - Multi-head attention explained

3. **[Positional Encoding](./positional-encoding.md)**
   - Why transformers need position information
   - Sinusoidal encoding vs learnable embeddings
   - How FundamentaLLM implements it

### Intermediate: "How Do We Train Them?"

4. **[Tokenization](./tokenization.md)**
   - Converting text to numbers
   - Character vs word vs subword tokenization
   - Why FundamentaLLM uses character-level

5. **[Language Modeling](./language-modeling.md)**
   - The core task: predict next token
   - Causal masking for autoregressive generation
   - How loss functions guide learning

6. **[Loss Functions](./losses.md)**
   - Cross-entropy for classification
   - Why it works for language modeling
   - Alternative loss functions

### Advanced: "Optimizing Training"

7. **[Optimization Algorithms](./optimization.md)**
   - Gradient descent and variants (SGD, Adam)
   - Why Adam is popular
   - How to choose optimizers

8. **[Learning Rate Scheduling](./scheduling.md)**
   - Constant vs adaptive learning rates
   - Warmup strategies
   - Common schedules

### Expert: "Implementing Efficiently"

9. **[Autoregressive Generation](./autoregressive.md)**
   - Generating tokens one-by-one
   - Temperature and top-k sampling
   - Beam search alternatives

10. **[Embeddings](./embeddings.md)**
    - Token embeddings
    - Positional embeddings
    - Why dimensionality matters

## Concept Map

```
┌─────────────────────────────────────────────┐
│           LANGUAGE MODELING                 │
│  (Predict next token given previous ones)   │
└──────┬──────────────────────────────────────┘
       │
       ├─ TOKENIZATION ──────────┐
       │  (Text → Numbers)       │
       │                         │
       ├─ EMBEDDINGS ────────────┤
       │  (Numbers → Vectors)    │
       │                         ├─→ TRANSFORMER
       ├─ POSITIONAL ENCODING ──┤   (Process &
       │  (Add position info)    │    combine
       │                         │    information)
       └─ ATTENTION ────────────┘
          (Focus on relevant parts)
       
       │
       ├─→ LOSS COMPUTATION
       │   (Measure error)
       │
       ├─→ OPTIMIZATION
       │   (Update weights)
       │
       └─→ GENERATION
           (Create new text)
```

## Key Insights

### Why Transformers?

Before transformers, recurrent models (RNNs, LSTMs) processed sequences step-by-step:
- Sequential processing = slow
- Hard to parallelize
- Information loss over long sequences

Transformers introduced **attention**:
- Process all tokens in parallel
- Each token can "see" every other token
- Better long-range dependencies
- Much faster to train

### Why Character-Level Tokenization?

```
Word-level:     ["Hello", "world"] → Fixed vocabulary
Character-level: ["H", "e", "l", "l", "o"] → 256 ASCII chars

Pros:
- No vocabulary management
- Handles any Unicode text (emojis, special chars, other languages)
- Simpler, more transparent

Cons:
- Longer sequences
- Harder to learn (more steps to read a word)
```

FundamentaLLM uses character-level for educational clarity and generality.

### Why Causal Masking?

In language modeling, we can't "cheat" by looking at future tokens:

```
Position: 1 2 3 4 5
Tokens:   T h e   c
Target:     h e   c a

Causal mask prevents position 1 from seeing positions 2-5
This forces autoregressive generation (left to right)
```

## How This Relates to Real LLMs

FundamentaLLM implements the **same core concepts** as:
- GPT (OpenAI)
- Claude (Anthropic)
- Llama (Meta)
- Gemini (Google)

The main differences in production models:
- **Scale:** 7B-100B+ parameters vs our ~1M
- **Training data:** Terabytes of text vs small datasets
- **Optimization:** Advanced techniques like rotary embeddings
- **Safety:** RLHF, alignment training
- **Engineering:** Inference optimization, serving infrastructure

But the transformer architecture is the same.

## Navigation

Start with your interest:

- **"Show me the math"** → [Transformer Architecture](./transformers.md)
- **"How do we train?"** → [Language Modeling](./language-modeling.md)
- **"What's attention?"** → [Attention Mechanism](./attention.md)
- **"Why these choices?"** → Individual concept pages
- **"Let me code it"** → [Modules](../modules/overview.md)
- **"Walk me through it"** → [Tutorials](../tutorials/first-model.md)

Each concept page includes:
- **Intuition:** High-level explanation
- **Mathematics:** Formal definitions
- **Implementation:** How FundamentaLLM does it
- **Why it matters:** Practical implications
- **Further reading:** Links to papers and resources
