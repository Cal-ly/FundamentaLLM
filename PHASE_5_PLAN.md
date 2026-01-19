# Phase 5: Generation & Evaluation

**Objective:** Implement text generation with various sampling strategies and model evaluation metrics.

**Status:** Planning

**Dependencies:** Phase 1 (Core Infrastructure) ✅, Phase 2 (Data Pipeline) ✅, Phase 3 (Model Architecture) ✅, Phase 4 (Training System) ✅

**Estimated Timeline:** 2-3 days

---

## Overview

Phase 5 adds inference capabilities to FundamentaLLM:
- Text generation from trained models
- Multiple sampling strategies (greedy, temperature, top-k, top-p)
- Generation constraints and stopping criteria
- Batch generation
- Model evaluation with standard metrics
- Inference utilities
- Benchmark datasets

This phase turns a trained model into a usable system.

---

## Files to Create

### Core Generation

```
src/fundamentallm/generation/
├── __init__.py                     # Generation module exports
├── generator.py                    # TextGenerator class
├── sampling.py                     # Sampling strategies
└── constraints.py                  # Generation constraints
```

### Evaluation

```
src/fundamentallm/evaluation/
├── __init__.py                     # Evaluation module exports
├── evaluator.py                    # ModelEvaluator class
└── benchmarks.py                   # Benchmark utilities
```

### Testing

```
tests/
├── unit/
│   ├── test_sampling.py            # Sampling tests
│   ├── test_generation.py          # Generation tests
│   └── test_evaluation.py          # Evaluation tests
└── integration/
    └── test_generation_pipeline.py # End-to-end generation
```

---

## Detailed Tasks

### Task 5.1: Sampling Strategies

**Objective:** Implement various text generation sampling methods

**File:** `src/fundamentallm/generation/sampling.py`

**Class: Sampler (Abstract)**

```python
class Sampler(ABC):
    """Base class for sampling strategies."""
    
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token from logits.
        
        Args:
            logits: [batch, vocab_size] or [vocab_size]
        
        Returns:
            Sampled token IDs
        """
        pass
```

**Sampler 1: GreedySampler**

Takes argmax (most probable token)

```python
class GreedySampler(Sampler):
    """Greedy decoding - always pick most likely token."""
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Return argmax tokens."""
        return torch.argmax(logits, dim=-1)
```

Pros: Deterministic, fast
Cons: Often repetitive, poor quality

**Sampler 2: TemperatureSampler**

Scale logits by temperature before softmax

```python
class TemperatureSampler(Sampler):
    """Temperature-scaled sampling."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Lower = more focused, Higher = more random
                        1.0 = no scaling
                        0.5 = sharper distribution
                        2.0 = flatter distribution
        """
        assert temperature > 0.0
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample with temperature scaling."""
        # Scale logits
        scaled_logits = logits / self.temperature
        
        # Convert to probabilities
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Sample from distribution
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.squeeze(-1)
```

Temperature effects:
- T→0: Greedy (sharp distribution)
- T=1: Normal softmax
- T→∞: Uniform (all tokens equally likely)

**Sampler 3: TopKSampler**

Only sample from top-k most likely tokens

```python
class TopKSampler(Sampler):
    """Top-k sampling - only consider k most likely tokens."""
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        assert k > 0
        assert temperature > 0.0
        self.k = k
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from top-k."""
        # Get top-k values
        top_k_vals, top_k_idx = torch.topk(logits, self.k, dim=-1)
        
        # Suppress non-top-k
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(-1, top_k_idx, top_k_vals)
        
        # Temperature scaling
        scaled_logits = logits_filtered / self.temperature
        
        # Sample
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.squeeze(-1)
```

Effects:
- k=1: Greedy
- k=10: Conservative (high quality)
- k=vocab_size: All tokens allowed

**Sampler 4: TopPSampler (Nucleus)**

Include tokens until cumulative probability reaches p

```python
class TopPSampler(Sampler):
    """Top-p (nucleus) sampling - include tokens until p cumulative prob."""
    
    def __init__(self, p: float = 0.95, temperature: float = 1.0):
        assert 0.0 < p <= 1.0
        assert temperature > 0.0
        self.p = p
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from top-p."""
        # Temperature scaling
        scaled_logits = logits / self.temperature
        
        # Get probabilities
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Sort by probability descending
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        
        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff (first index where cumsum > p)
        cutoff_mask = cumsum_probs > self.p
        
        # Shift mask to include the token that crosses p
        cutoff_mask[..., 0] = False  # Never exclude most likely token
        sorted_probs[cutoff_mask] = 0.0
        
        # Renormalize
        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
        
        # Sample
        sampled = torch.multinomial(probs_filtered, num_samples=1)
        return sampled.squeeze(-1)
```

Effects:
- p=0.5: Conservative
- p=0.95: Typical
- p=1.0: All tokens allowed

**Sampler Factory:**

```python
def create_sampler(
    strategy: str,
    **kwargs
) -> Sampler:
    """Create sampler from strategy name."""
    if strategy == "greedy":
        return GreedySampler()
    elif strategy == "temperature":
        return TemperatureSampler(kwargs.get("temperature", 1.0))
    elif strategy == "top_k":
        return TopKSampler(
            k=kwargs.get("k", 50),
            temperature=kwargs.get("temperature", 1.0)
        )
    elif strategy == "top_p":
        return TopPSampler(
            p=kwargs.get("p", 0.95),
            temperature=kwargs.get("temperature", 1.0)
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
```

**Success Criteria:**
- ✅ All samplers produce valid token IDs
- ✅ Greedy gives argmax
- ✅ Temperature scales correctly
- ✅ Top-k masks correctly
- ✅ Top-p cumulative probability correct

---

### Task 5.2: Text Generator

**Objective:** Main class for generating text

**File:** `src/fundamentallm/generation/generator.py`

**Class: TextGenerator**

```python
class TextGenerator:
    """Generate text from trained model."""
    
    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        device: str = "cuda",
        sampler: Optional[Sampler] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.sampler = sampler or GreedySampler()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cuda"
    ) -> "TextGenerator":
        """Load model and tokenizer from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Recreate model
        config = checkpoint["config"]
        model = ModelRegistry.create("transformer", config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load tokenizer
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
        tokenizer = BaseTokenizer.load(tokenizer_path)
        
        return cls(model, tokenizer, device)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        verbose: bool = False
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Initial prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k constraint (if set)
            top_p: Top-p constraint (if set)
            stop_sequences: Sequences that trigger stop
            verbose: Print generation progress
        
        Returns:
            Generated text (including prompt)
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], device=self.device, dtype=torch.long)
        
        # Generate
        generated_tokens = []
        
        for i in range(max_tokens):
            # Forward pass
            logits = self.model(input_ids)
            next_logits = logits[0, -1, :]  # Get last token logits
            
            # Select sampler based on parameters
            if top_k is not None:
                sampler = TopKSampler(top_k, temperature)
            elif top_p is not None:
                sampler = TopPSampler(top_p, temperature)
            else:
                sampler = TemperatureSampler(temperature)
            
            # Sample
            next_token = sampler.sample(next_logits)
            generated_tokens.append(next_token.item())
            
            # Append to input for next iteration
            input_ids = torch.cat([
                input_ids,
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
            
            # Check stopping criteria
            if stop_sequences:
                current_text = self.tokenizer.decode(generated_tokens)
                for stop_seq in stop_sequences:
                    if stop_seq in current_text:
                        if verbose:
                            print(f"\nStopped at: {stop_seq}")
                        return prompt + current_text
            
            if verbose and (i + 1) % 50 == 0:
                current_text = self.tokenizer.decode(generated_tokens)
                print(f"Generated {i+1} tokens...")
        
        # Decode and return
        generated_text = self.tokenizer.decode(generated_tokens)
        return prompt + generated_text
    
    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 200,
        **kwargs
    ) -> List[str]:
        """Generate from multiple prompts."""
        return [self.generate(prompt, max_tokens, **kwargs) for prompt in prompts]
```

**Usage Example:**

```python
# Load from checkpoint
generator = TextGenerator.from_checkpoint("checkpoints/best.pt")

# Generate text
text = generator.generate(
    prompt="The future of AI",
    max_tokens=100,
    temperature=0.8,
    top_p=0.95
)

print(text)
```

**Success Criteria:**
- ✅ Can generate text from prompt
- ✅ Sampling strategies work
- ✅ Stop sequences honored
- ✅ Batch generation works
- ✅ Can load from checkpoint

---

### Task 5.3: Model Evaluator

**Objective:** Evaluate model on test sets

**File:** `src/fundamentallm/evaluation/evaluator.py`

**Class: ModelEvaluator**

```python
class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader with (input_ids, target_ids) pairs
            return_predictions: Also return model predictions
        
        Returns:
            Dict with metrics (loss, perplexity, accuracy)
        """
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        all_predictions = [] if return_predictions else None
        all_targets = [] if return_predictions else None
        
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            logits = self.model(inputs)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # Metrics
            total_loss += loss.item() * inputs.size(0)
            
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            
            if return_predictions:
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / (len(data_loader) * inputs.size(0))
        accuracy = total_correct / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        results = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy
        }
        
        if return_predictions:
            results["predictions"] = torch.cat(all_predictions)
            results["targets"] = torch.cat(all_targets)
        
        return results
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cuda"
    ) -> "ModelEvaluator":
        """Load from checkpoint."""
        # Similar to TextGenerator.from_checkpoint()
        # ... implementation ...
        pass
```

**Success Criteria:**
- ✅ Can evaluate on dataset
- ✅ Metrics computed correctly
- ✅ Predictions returned if requested
- ✅ Handles different batch sizes

---

### Task 5.4: Generation Tests

**Objective:** Test sampling and generation

**File:** `tests/unit/test_sampling.py`

Tests:
- ✅ All samplers produce valid tokens
- ✅ Greedy gives argmax
- ✅ Temperature affects distribution
- ✅ Top-k masking works
- ✅ Top-p cumulative probability correct

**File:** `tests/unit/test_generation.py`

Tests:
- ✅ Can generate from prompt
- ✅ Generation respects max_tokens
- ✅ Stop sequences work
- ✅ Batch generation works
- ✅ Output is valid text

**File:** `tests/integration/test_generation_pipeline.py`

Tests:
- ✅ End-to-end generation works
- ✅ Can load checkpoint and generate
- ✅ Multiple generations different (with temperature)
- ✅ Generated tokens in vocab range

**Success Criteria:**
- ✅ All tests pass
- ✅ Coverage > 85%

---

### Task 5.5: Documentation

**Objective:** Document generation and evaluation

**Docstrings:**
- All classes documented
- Type hints complete
- Usage examples provided

**Example Usage:**

```python
# Generate text
generator = TextGenerator.from_checkpoint("model.pt")
text = generator.generate("Once upon a time", max_tokens=100)

# Evaluate model
evaluator = ModelEvaluator.from_checkpoint("model.pt")
metrics = evaluator.evaluate(test_loader)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

---

## Implementation Checklist

- [ ] Create sampling strategies (Task 5.1)
  - [ ] GreedySampler
  - [ ] TemperatureSampler
  - [ ] TopKSampler
  - [ ] TopPSampler
  - [ ] Sampler factory
- [ ] Create TextGenerator (Task 5.2)
  - [ ] generate() method
  - [ ] batch_generate() method
  - [ ] from_checkpoint() classmethod
- [ ] Create ModelEvaluator (Task 5.3)
- [ ] Create tests (Task 5.4)
- [ ] Add documentation (Task 5.5)

---

## Success Criteria for Phase 5

1. **Sampling Strategies**
   - ✅ All strategies produce valid tokens
   - ✅ Quality increases: greedy < temperature < top-k/top-p

2. **Text Generation**
   - ✅ Can generate from prompt
   - ✅ Supports multiple sampling strategies
   - ✅ Stop sequences work
   - ✅ Batch generation works

3. **Evaluation**
   - ✅ Can evaluate on test sets
   - ✅ Metrics computed correctly
   - ✅ Predictions available

4. **Testing**
   - ✅ All components tested
   - ✅ Coverage > 85%

---

## Next Phase

Phase 5 is complete when generation and evaluation work end-to-end.

Phase 6 will add CLI and interactive interface.

---

## Notes

- Temperature sampling is most important for quality
- Top-p (nucleus) often better than top-k in practice
- Consider generation speed vs quality tradeoff
- Document sampling hyperparameters in generated text (optional)
