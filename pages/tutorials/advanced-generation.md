# Advanced Generation

Master text generation with advanced sampling strategies and control techniques.

## Sampling Methods

### Temperature Sampling

**Control randomness of generation.**

```python
def sample_with_temperature(logits, temperature=1.0):
    """
    Args:
        logits: (vocab_size,) raw model outputs
        temperature: float, controls randomness
            - Low (0.1-0.5): Focused, deterministic
            - Medium (0.8-1.2): Balanced
            - High (1.5-2.0): Creative, random
    
    Returns:
        token_id: sampled token
    """
    # Scale logits
    scaled_logits = logits / temperature
    
    # Compute probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample
    token_id = torch.multinomial(probs, num_samples=1)
    
    return token_id.item()
```

**Temperature effects:**

```bash
# Temperature = 0.3 (conservative)
$ fundamentallm generate model.pt --prompt "The cat" --temperature 0.3
"The cat sat on the mat and looked around."

# Temperature = 0.8 (balanced)
$ fundamentallm generate model.pt --prompt "The cat" --temperature 0.8
"The cat wandered through the garden, chasing butterflies."

# Temperature = 1.5 (creative)
$ fundamentallm generate model.pt --prompt "The cat" --temperature 1.5
"The cat mysteriously vanished into starlight, leaving whispers."
```

**When to use:**
- Low: Formal text, factual content
- Medium: General use, creative writing
- High: Brainstorming, exploration

### Top-k Sampling

**Restrict to k most likely tokens.**

```python
def top_k_sampling(logits, k=50, temperature=1.0):
    """
    Sample from top-k most likely tokens.
    
    Args:
        logits: (vocab_size,) raw model outputs
        k: number of top tokens to consider
        temperature: sampling temperature
    
    Returns:
        token_id: sampled token
    """
    # Scale logits
    logits = logits / temperature
    
    # Get top-k
    top_k_logits, top_k_indices = torch.topk(logits, k=k)
    
    # Compute probabilities over top-k
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from top-k
    idx = torch.multinomial(probs, num_samples=1)
    token_id = top_k_indices[idx]
    
    return token_id.item()
```

**Example:**

```python
# Original distribution
probs = [0.4, 0.3, 0.15, 0.08, 0.05, 0.02, ...]  # 256 tokens

# Top-k=5
top_k_probs = [0.4, 0.3, 0.15, 0.08, 0.05]
# Renormalize: [0.41, 0.31, 0.15, 0.08, 0.05]

# Only sample from these 5 tokens
```

**When to use:**
- k=1: Greedy decoding (deterministic)
- k=10-20: Very focused
- k=50: Balanced (default)
- k=100+: More diverse

**Usage:**

```bash
fundamentallm generate model.pt \ \
    --prompt "Once upon a time" \ \
    --top-k 40 \ \
    --temperature 0.8
```

### Top-p (Nucleus) Sampling

**Sample from smallest set with cumulative probability ≥ p.**

```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus sampling: sample from top tokens that sum to p probability.
    
    Args:
        logits: (vocab_size,) raw model outputs
        p: cumulative probability threshold (0.0-1.0)
        temperature: sampling temperature
    
    Returns:
        token_id: sampled token
    """
    # Scale logits
    logits = logits / temperature
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    # Keep at least one token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[0] = False  # Always keep top token
    
    # Filter
    filtered_probs = sorted_probs.clone()
    filtered_probs[sorted_indices_to_remove] = 0.0
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    # Sample
    idx = torch.multinomial(filtered_probs, num_samples=1)
    token_id = sorted_indices[idx]
    
    return token_id.item()
```

**Example:**

```python
# Model is confident
probs = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02, ...]
# Top-p=0.9: Use [0.5, 0.3, 0.1] (sum=0.9)

# Model is uncertain
probs = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, ...]
# Top-p=0.9: Use [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.08, 0.07] (many tokens!)
```

**Adaptive:** Cutoff changes based on confidence.

**When to use:**
- p=0.9: Balanced (default)
- p=0.95: More diverse
- p=0.85: More focused

**Usage:**

```bash
fundamentallm generate model.pt \ \
    --prompt "Once upon a time" \ \
    --top-p 0.9 \ \
    --temperature 0.8
```

### Combined: Top-k + Top-p

**Best practice: Use both!**

```python
def combined_sampling(logits, temperature=1.0, top_k=50, top_p=0.9):
    """
    Apply both top-k and top-p filtering.
    """
    logits = logits / temperature
    
    # Apply top-k
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Apply top-p
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum()
    
    # Sample
    token_id = torch.multinomial(probs, num_samples=1)
    return token_id.item()
```

**Usage:**

```bash
fundamentallm generate model.pt \ \
    --prompt "Once upon a time" \ \
    --temperature 0.8 \ \
    --top-k 50 \ \
    --top-p 0.9
```

## Repetition Control

### Repetition Penalty

**Penalize tokens that have appeared recently.**

```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    """
    Reduce probability of recently generated tokens.
    
    Args:
        logits: (vocab_size,) current logits
        generated_tokens: list of previously generated token IDs
        penalty: multiplicative penalty (> 1.0)
    
    Returns:
        logits: penalized logits
    """
    for token_id in set(generated_tokens):
        # Penalize tokens that have appeared
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    
    return logits
```

**Example:**

```python
# Without penalty
"The cat sat on the mat. The cat sat on the mat. The cat..."

# With penalty=1.2
"The cat sat on the mat. She looked around curiously, then..."
```

**Usage:**

```bash
fundamentallm generate model.pt \ \
    --prompt "The cat" \ \
    --repetition-penalty 1.2
```

### Frequency Penalty

**Penalize based on frequency of occurrence.**

```python
from collections import Counter

def apply_frequency_penalty(logits, generated_tokens, penalty=0.1):
    """
    Penalize tokens based on how often they've appeared.
    
    Args:
        logits: (vocab_size,) current logits
        generated_tokens: list of previously generated token IDs
        penalty: penalty per occurrence
    
    Returns:
        logits: penalized logits
    """
    token_counts = Counter(generated_tokens)
    
    for token_id, count in token_counts.items():
        logits[token_id] -= penalty * count
    
    return logits
```

**More aggressive** than repetition penalty.

### Presence Penalty

**Penalize if token appeared at all (binary).**

```python
def apply_presence_penalty(logits, generated_tokens, penalty=0.5):
    """
    Penalize tokens that have appeared (regardless of frequency).
    
    Args:
        logits: (vocab_size,) current logits
        generated_tokens: list of previously generated token IDs
        penalty: fixed penalty for presence
    
    Returns:
        logits: penalized logits
    """
    appeared_tokens = set(generated_tokens)
    
    for token_id in appeared_tokens:
        logits[token_id] -= penalty
    
    return logits
```

**Encourages diversity** without over-penalizing.

## Beam Search

**Explore multiple candidates simultaneously.**

### Algorithm

```python
def beam_search(model, prompt_tokens, beam_width=5, max_length=100):
    """
    Beam search decoding.
    
    Args:
        model: language model
        prompt_tokens: initial tokens
        beam_width: number of beams to maintain
        max_length: maximum sequence length
    
    Returns:
        best_sequence: highest-scoring sequence
    """
    # Initialize beams: (sequence, score)
    beams = [(prompt_tokens, 0.0)]
    
    for _ in range(max_length - len(prompt_tokens)):
        candidates = []
        
        for sequence, score in beams:
            # Get next token probabilities
            logits = model(torch.tensor([sequence]))
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            
            # Get top-k candidates
            top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width)
            
            for log_prob, token_id in zip(top_k_log_probs, top_k_indices):
                new_sequence = sequence + [token_id.item()]
                new_score = score + log_prob.item()
                candidates.append((new_sequence, new_score))
        
        # Keep top beam_width sequences
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all beams ended
        if all(seq[-1] == END_TOKEN for seq, _ in beams):
            break
    
    # Return best sequence
    best_sequence, best_score = beams[0]
    return best_sequence
```

**Pros:**
- More coherent output
- Finds high-probability sequences

**Cons:**
- Slower (beam_width × slower)
- Can be repetitive
- Less creative

**Usage:**

```bash
fundamentallm generate model.pt \ \
    --prompt "Once upon a time" \ \
    --beam-width 5
```

### Length Normalization

**Prevent bias toward short sequences:**

```python
def length_normalized_score(sequence, score, alpha=0.6):
    """
    Normalize score by sequence length.
    
    Args:
        sequence: list of tokens
        score: log-probability sum
        alpha: length penalty (0=no penalty, 1=full normalization)
    
    Returns:
        normalized_score: length-adjusted score
    """
    length_penalty = ((5 + len(sequence)) / 6) ** alpha
    return score / length_penalty
```

## Constrained Generation

### Prefix Constraints

**Force generation to start with specific text:**

```python
def generate_with_prefix(model, prefix, max_length=100):
    """Generate text that must start with prefix."""
    # Tokenize prefix
    tokens = tokenizer.encode(prefix)
    
    # Generate continuation
    for _ in range(max_length - len(tokens)):
        logits = model(torch.tensor([tokens]))
        next_token = sample(logits[0, -1])
        tokens.append(next_token)
        
        if next_token == END_TOKEN:
            break
    
    return tokenizer.decode(tokens)
```

### Banned Tokens

**Prevent generation of specific tokens:**

```python
def generate_with_banned_tokens(model, prompt, banned_tokens, max_length=100):
    """Generate while avoiding banned tokens."""
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_length - len(tokens)):
        logits = model(torch.tensor([tokens]))
        
        # Set banned tokens to -inf
        for token_id in banned_tokens:
            logits[0, -1, token_id] = float('-inf')
        
        next_token = sample(logits[0, -1])
        tokens.append(next_token)
    
    return tokenizer.decode(tokens)
```

**Example:**

```python
# Ban punctuation
banned = tokenizer.encode(".,!?;:")
generate_with_banned_tokens(model, "Hello", banned)
# → "Hello world this is a test"  (no punctuation)
```

### Format Constraints

**Enforce specific formats:**

```python
def generate_dialogue(model, max_turns=5):
    """Generate dialogue with alternating speakers."""
    tokens = []
    
    for turn in range(max_turns):
        # Add speaker marker
        speaker = "Alice: " if turn % 2 == 0 else "Bob: "
        tokens.extend(tokenizer.encode(speaker))
        
        # Generate utterance (stop at newline)
        while True:
            logits = model(torch.tensor([tokens]))
            next_token = sample(logits[0, -1])
            tokens.append(next_token)
            
            if tokenizer.decode([next_token]) == '\n':
                break
    
    return tokenizer.decode(tokens)
```

## Interactive Generation

### Streaming Output

**Generate token-by-token:**

```python
def generate_streaming(model, prompt, max_length=100):
    """
    Generate and print tokens as they're produced.
    """
    tokens = tokenizer.encode(prompt)
    print(prompt, end='', flush=True)
    
    for _ in range(max_length - len(tokens)):
        logits = model(torch.tensor([tokens]))
        next_token = sample(logits[0, -1])
        tokens.append(next_token)
        
        # Print immediately
        char = tokenizer.decode([next_token])
        print(char, end='', flush=True)
        
        if next_token == END_TOKEN:
            break
    
    print()  # Newline at end
```

### User-guided Generation

```python
def interactive_generation(model):
    """Interactive generation with user control."""
    print("Interactive Generation (type 'quit' to exit)")
    
    tokens = []
    
    while True:
        # Generate next token
        if tokens:
            logits = model(torch.tensor([tokens]))
            probs = F.softmax(logits[0, -1], dim=-1)
            
            # Show top-5 options
            top_5_probs, top_5_indices = torch.topk(probs, 5)
            print("\nTop 5 options:")
            for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices)):
                char = tokenizer.decode([idx.item()])
                print(f"  {i+1}. '{char}' ({prob:.2%})")
            
            # User choice
            choice = input("Choose (1-5, or type custom, or 'quit'): ")
            
            if choice == 'quit':
                break
            elif choice.isdigit() and 1 <= int(choice) <= 5:
                next_token = top_5_indices[int(choice)-1].item()
            else:
                next_token = tokenizer.encode(choice)[0]
            
            tokens.append(next_token)
            print(f"Current: {tokenizer.decode(tokens)}")
        else:
            # Initial prompt
            prompt = input("Enter prompt: ")
            tokens = tokenizer.encode(prompt)
```

## Advanced Techniques

### Speculative Decoding

**Generate multiple tokens in parallel (approximation):**

```python
def speculative_decoding(model, draft_model, prompt, k=4):
    """
    Use draft model to propose k tokens, verify with main model.
    Faster generation with same quality.
    """
    tokens = tokenizer.encode(prompt)
    
    while len(tokens) < max_length:
        # Draft model proposes k tokens
        draft_tokens = []
        draft_probs = []
        
        current_tokens = tokens[:]
        for _ in range(k):
            logits = draft_model(torch.tensor([current_tokens]))
            prob = F.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(prob, 1).item()
            
            draft_tokens.append(next_token)
            draft_probs.append(prob[next_token].item())
            current_tokens.append(next_token)
        
        # Main model verifies
        full_logits = model(torch.tensor([tokens + draft_tokens]))
        
        # Accept tokens until mismatch
        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            real_prob = F.softmax(full_logits[0, len(tokens) + i - 1], dim=-1)
            accept_prob = min(1.0, real_prob[draft_token].item() / draft_prob)
            
            if random.random() < accept_prob:
                tokens.append(draft_token)
            else:
                # Sample from adjusted distribution
                adjusted_probs = torch.max(real_prob - draft_prob, torch.zeros_like(real_prob))
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                next_token = torch.multinomial(adjusted_probs, 1).item()
                tokens.append(next_token)
                break
    
    return tokenizer.decode(tokens)
```

**Speed up:** 2-3× faster with small accuracy trade-off.

## Practical Tips

### 1. Start with Defaults

```bash
# Good starting point
--temperature 0.8 \
--top-k 50 \
--top-p 0.9
```

### 2. Adjust for Use Case

**Creative writing:**
```bash
--temperature 1.0 \
--top-p 0.95 \
--repetition-penalty 1.1
```

**Factual/formal:**
```bash
--temperature 0.5 \
--top-k 20 \
--top-p 0.85
```

**Diverse outputs:**
```bash
--temperature 1.2 \
--top-p 0.95 \
--repetition-penalty 1.3
```

### 3. Multiple Samples

Generate several, pick best:

```bash
fundamentallm generate model.pt \ \
    --prompt "Once upon a time" \ \
    --num-samples 5 \ \
    --temperature 0.9
```

### 4. Control Length

```bash
--max-length 200        # Hard limit
--min-length 50         # Minimum before allowing end
--stop-tokens "." "!" "?"  # Natural stopping points
```

## Further Reading

- [Generation Module](../modules/generation.md) - Implementation
- [Sampling Concepts](../concepts/autoregressive.md) - Theory
- "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) - Top-p paper
- "Beam Search Strategies" (Wiseman & Rush, 2016)

## Next Steps

- [Custom Datasets](./custom-datasets.md) - Train on your data
- [Evaluation](../guide/evaluation.md) - Measure quality
- [Interactive Mode](../guide/cli-overview.md#interactive) - Real-time generation
