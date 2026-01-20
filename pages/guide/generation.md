# Text Generation

Learn how to generate text from trained models and control the generation process.

## Quick Start

```bash
# Basic generation
fundamentallm generate my_model/final_model.pt \
    --prompt "Once upon a time"

# Interactive mode
fundamentallm generate my_model/final_model.pt --interactive
```

## How Generation Works

### Autoregressive Process

Language models generate **one token at a time**, using previous tokens as context:

```
1. Start:  "Once"
2. Predict: "upon"      → "Once upon"
3. Predict: "a"         → "Once upon a"  
4. Predict: "time"      → "Once upon a time"
5. Predict: "there"     → "Once upon a time there"
6. Continue...
```

Each prediction is conditioned on all previous tokens.

### The Generation Loop

```python
def generate(model, prompt, max_tokens=100):
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        # Get predictions for next token
        logits = model(tokens)
        next_token_logits = logits[-1]  # Last position
        
        # Sample next token
        next_token = sample(next_token_logits)
        
        # Append to sequence
        tokens.append(next_token)
        
        # Stop if end token
        if next_token == END_TOKEN:
            break
    
    return tokenizer.decode(tokens)
```

## Sampling Strategies

### 1. Greedy Sampling

**What:** Always pick the most likely token.

```python
next_token = torch.argmax(logits)
```

**Pros:**
- ✅ Deterministic (same output every time)
- ✅ Fast

**Cons:**
- ❌ Boring and repetitive
- ❌ Gets stuck in loops

**Example:**
```
Prompt: "The cat"
Greedy: "The cat is a cat is a cat is a cat..."  ← loops!
```

**Usage:**
```bash
--temperature 0.0  # Greedy (effectively)
```

### 2. Temperature Sampling

**What:** Sample from probability distribution with temperature control.

$$P(x_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

```python
# Apply temperature
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits, dim=-1)

# Sample
next_token = torch.multinomial(probs, num_samples=1)
```

**Temperature values:**

**T = 0.1 (Low - Focused)**
```
Prompt: "The cat"
Output: "The cat sat quietly on the windowsill."
```
- Very predictable
- Safe, grammatical
- Less creative

**T = 1.0 (Standard - Balanced)**
```
Prompt: "The cat"
Output: "The cat explored the mysterious garden."
```
- Natural probability distribution
- Good balance
- **FundamentaLLM default**

**T = 2.0 (High - Creative)**
```
Prompt: "The cat"
Output: "The cat danced beneath forgotten stars."
```
- More random
- Creative but less coherent
- Can be nonsensical

**Usage:**
```bash
# Conservative
--temperature 0.5

# Balanced
--temperature 1.0

# Creative
--temperature 1.5
```

### 3. Top-k Sampling

**What:** Only sample from the k most likely tokens.

```python
# Get top k tokens
top_k_logits, top_k_indices = torch.topk(logits, k=50)

# Zero out others
filtered_logits = torch.full_like(logits, float('-inf'))
filtered_logits[top_k_indices] = top_k_logits

# Sample from top-k
probs = F.softmax(filtered_logits / temperature, dim=-1)
next_token = torch.multinomial(probs, 1)
```

**Effect:** Prevents very unlikely tokens, reducing nonsense.

**K values:**

```bash
# Very focused (top 10 tokens only)
--top-k 10

# Balanced (top 50 tokens)
--top-k 50

# Diverse (top 100 tokens)
--top-k 100
```

**Example:**
```
Without top-k: "The cat $#@ qwerty..."  ← random chars appear
With top-k=50: "The cat walked slowly..."  ← only plausible words
```

### 4. Top-p (Nucleus) Sampling

**What:** Sample from smallest set of tokens whose cumulative probability exceeds p.

```python
# Sort by probability
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

# Remove tokens with cumulative probability > p
remove_indices = cumulative_probs > p
remove_indices[1:] = remove_indices[:-1].clone()
remove_indices[0] = False

# Filter and sample
filtered_probs = probs.clone()
filtered_probs[sorted_indices[remove_indices]] = 0
next_token = torch.multinomial(filtered_probs, 1)
```

**P values:**

```bash
# Focused (only very likely tokens)
--top-p 0.8

# Balanced
--top-p 0.9

# Diverse (most tokens)
--top-p 0.95
```

**Advantage over top-k:** Adaptive - adjusts number of candidates based on confidence distribution.

### Combining Strategies

Best practice: **Temperature + Top-p**

```bash
fundamentallm generate model.pt \
    --prompt "The wizard" \
    --temperature 0.8 \
    --top-p 0.9
```

This gives creative but coherent generation.

## Generation Parameters

### Max Tokens (`--max-tokens`)

```bash
# Short generation
--max-tokens 50

# Medium
--max-tokens 200

# Long
--max-tokens 500
```

Limits generation length.

### Number of Samples (`--num-samples`)

```bash
# Generate 5 different outputs
--num-samples 5
```

Useful for exploring different continuations.

### Seed (`--seed`)

```bash
# Reproducible generation
--seed 42
```

Same seed + same parameters = same output.

## Interactive Mode

Most flexible way to explore your model:

```bash
fundamentallm generate my_model/final_model.pt --interactive
```

### Available Commands

```
> Your prompt here          # Generate from prompt
> /help                     # Show commands
> /set temperature=0.8      # Change parameter
> /set top-k=50            # Change parameter
> /status                   # Show current settings
> /clear                    # Clear screen
> /quit                     # Exit
```

### Example Session

```
╭──────────────────────────────────────────────────╮
│ FundamentaLLM Interactive Mode                   │
│ Type /help for commands, /quit to exit          │
╰──────────────────────────────────────────────────╯

> Once upon a time
Once upon a time, in a faraway kingdom, there lived
a brave knight who sought adventure...

> /set temperature=0.5
Updated: temperature=0.5

> Once upon a time
Once upon a time, there was a small village nestled
in the mountains...

> /set temperature=1.5
Updated: temperature=1.5

> Once upon a time
Once upon a time, beneath crystalline moons, the
ancient prophecy whispered through cosmic winds...

> /quit
Goodbye!
```

## Generation Strategies

### Creative Writing

```bash
fundamentallm generate model.pt \
    --prompt "Write a short story:" \
    --temperature 1.2 \
    --top-p 0.9 \
    --max-tokens 500
```

High temperature for creativity.

### Code Generation

```bash
fundamentallm generate model.pt \
    --prompt "def fibonacci(n):" \
    --temperature 0.3 \
    --top-p 0.95 \
    --max-tokens 200
```

Low temperature for correctness.

### Completion

```bash
fundamentallm generate model.pt \
    --prompt "The quick brown" \
    --temperature 0.7 \
    --max-tokens 50
```

Medium temperature for natural completion.

### Poetry

```bash
fundamentallm generate model.pt \
    --prompt "Roses are red," \
    --temperature 1.0 \
    --top-k 40 \
    --max-tokens 100
```

Balanced for structure + creativity.

## Common Patterns

### Batch Generation

```bash
# Generate multiple outputs to file
fundamentallm generate model.pt \
    --prompt "The ancient scroll revealed" \
    --num-samples 10 \
    --output-file generations.txt
```

### Scripted Generation

```python
from fundamentallm import Generator, load_model

model = load_model('my_model/final_model.pt')
generator = Generator(model, temperature=0.8)

prompts = [
    "The dragon",
    "In the forest",
    "A mysterious letter"
]

for prompt in prompts:
    output = generator.generate(prompt, max_tokens=100)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")
```

### Conditional Generation

```bash
# Style conditioning
--prompt "In Shakespeare's style:"

# Task conditioning
--prompt "Translate to French:"

# Format conditioning
--prompt "Q: What is AI? A:"
```

## Quality Control

### Repetition Penalty

Reduce repetitive outputs:

```bash
--repetition-penalty 1.2
```

Higher values discourage repeating tokens.

### Length Penalty

Encourage longer/shorter outputs:

```bash
--length-penalty 1.5  # Prefer longer
--length-penalty 0.5  # Prefer shorter
```

### Bad Words Filter

Block specific tokens:

```bash
--bad-words "profanity,offensive,word"
```

## Debugging Generation

### Check Model Quality

```bash
# Test with simple prompts
--prompt "The"
--prompt "Hello"
--prompt "Once upon a time"
```

Good model should produce coherent continuations.

### Inspect Probabilities

```bash
# Show top predictions
--show-probs
```

Outputs:
```
Next token probabilities:
  " " (space): 0.35
  "c": 0.25
  "w": 0.15
  ...
```

### Compare Temperatures

```bash
# Try different temperatures
for temp in 0.3 0.7 1.0 1.5; do
    fundamentallm generate model.pt \
        --prompt "Test" \
        --temperature $temp \
        --max-tokens 50
done
```

## Advanced Techniques

### Beam Search

Instead of sampling, keep top-k candidates:

```bash
--beam-size 5
```

More deterministic but slower.

### Constrained Generation

Force specific tokens or patterns:

```bash
--must-include "dragon,castle,knight"
```

### Guided Generation

Steer generation toward desired attributes:

```bash
--guide-attribute "positive_sentiment"
--guide-strength 2.0
```

## Performance Tips

### Faster Generation

```bash
# Use GPU
--device cuda

# Reduce max tokens
--max-tokens 100

# Use greedy (no sampling overhead)
--temperature 0.0
```

### Better Quality

```bash
# Multiple samples, pick best
--num-samples 5

# Combine temperature + top-p
--temperature 0.8 --top-p 0.9

# Use beam search
--beam-size 3
```

## Common Issues

### Repetitive Output

**Fix:**
```bash
--repetition-penalty 1.2
--temperature 1.0  # Increase from too low
```

### Nonsense Output

**Fix:**
```bash
--temperature 0.7  # Decrease from too high
--top-p 0.9        # Add nucleus sampling
--top-k 50         # Add top-k filtering
```

### Too Conservative

**Fix:**
```bash
--temperature 1.2  # Increase
--top-p 0.95       # Allow more diversity
```

### Wrong Style

**Cause:** Model not trained on that style  
**Fix:** Fine-tune on target style or use better prompting

## Examples by Use Case

### Story Generation
```bash
--prompt "Chapter 1:" \
--temperature 1.0 \
--top-p 0.9 \
--max-tokens 500
```

### Code Completion
```bash
--prompt "def sort_list(arr):" \
--temperature 0.3 \
--max-tokens 200
```

### Dialog
```bash
--prompt "User: Hello\nAssistant:" \
--temperature 0.7 \
--top-p 0.9
```

### Poetry
```bash
--prompt "A haiku about fall:" \
--temperature 1.1 \
--max-tokens 50
```

## Next Steps

- [Sampling](../concepts/autoregressive.md) - Theory behind sampling
- [CLI Overview](./cli-overview.md) - All generation commands
- [Generation Module](../modules/generation.md) - Implementation
- [Advanced Generation Tutorial](../tutorials/advanced-generation.md) - Deep dive
