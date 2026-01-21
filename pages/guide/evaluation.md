# Evaluation

Learn how to measure and improve your language model's performance.

## Quick Reference

```bash
# Evaluate on validation data
fundamentallm evaluate model.pt --data val.txt

# Generate samples
fundamentallm generate model.pt --prompt "Test" --num-samples 5

# Interactive testing
fundamentallm interactive model.pt
```

## Metrics

### Perplexity

**Most important metric for language models.**

**What it means:** How "surprised" the model is by the data.
- Lower = better
- Measures: exp(loss)

**Interpretation:**

| Perplexity | Quality | Meaning |
|------------|---------|---------|
| ~256 | Random | Guessing uniformly over vocab |
| 50-100 | Poor | Barely better than random |
| 20-50 | Weak | Learning some patterns |
| 10-20 | Decent | Reasonable performance |
| 3-10 | Good | Captures most patterns |
| 1.5-3 | Excellent | Very strong model |
| <1.5 | Suspicious | Likely overfitting |

**Example:**
```bash
$ fundamentallm evaluate model.pt --data val.txt
Perplexity: 4.23
Loss: 1.442

# Interpretation: Good model!
# Perplexity of 4.23 means model effectively has 4-5 choices per character
```

### Cross-Entropy Loss

**What it measures:** Average negative log-likelihood per token.

**Formula:**
$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$$

**Interpretation:**
- Lower = better
- Related to perplexity: $\text{Perplexity} = \exp(\text{Loss})$

**Character-level ranges:**
```
Loss > 3.0: Poor model
Loss 2.0-3.0: Weak model
Loss 1.5-2.0: Decent model
Loss 1.0-1.5: Good model
Loss < 1.0: Excellent (or overfit)
```

### Bits Per Character (BPC)

**What it measures:** Information theory metric.

**Formula:**
$$\text{BPC} = \frac{\text{Loss}}{\log(2)} = \text{Loss} \times 1.44$$

**Interpretation:**
- How many bits needed to encode each character
- Lower = better compression = better model

**Comparison:**
```
ASCII encoding: 8 bits/char
Good compression: ~2 bits/char
Strong model: 1-2 BPC
Excellent model: 0.5-1 BPC
```

## Quantitative Evaluation

### Validation Loss

**Most reliable metric.**

```bash
# During training, watch validation loss
Epoch 1: train_loss=3.45 val_loss=3.52
Epoch 5: train_loss=2.12 val_loss=2.18
Epoch 10: train_loss=1.67 val_loss=1.76
```

**Good signs:**
- ✅ Both decreasing
- ✅ Small gap between train and val
- ✅ Smooth convergence

**Bad signs:**
- ❌ Val loss increasing (overfitting)
- ❌ Large train/val gap (overfitting)
- ❌ Not decreasing (underfitting)

### Test Set Evaluation

**After training:**

```bash
# Evaluate on held-out test set
fundamentallm evaluate model.pt --data test.txt

# Results
Test Loss: 1.834
Test Perplexity: 6.26
```

**Important:** 
- Never train on test data
- Use for final evaluation only
- Should be similar to validation performance

### Learning Curves

**Plot metrics over training:**

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
train_loss = [3.45, 2.87, 2.34, 2.01, 1.76]
val_loss = [3.52, 2.91, 2.41, 2.12, 1.89]

plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**What to look for:**
- Convergence: Both curves flattening
- Overfitting: Train continues down, val plateaus/rises
- Underfitting: Both high and not improving

## Qualitative Evaluation

**Metrics don't tell the whole story.** Always check generated text!

### Sample Generation

```bash
# Generate multiple samples
fundamentallm generate model.pt \ \
    --prompt "Once upon a time" \ \
    --num-samples 5 \ \
    --max-length 200 \ \
    --temperature 0.8
```

**What to check:**
- Coherence: Does it make sense?
- Grammar: Proper syntax?
- Creativity: Diverse outputs?
- Consistency: Stays on topic?

### Temperature Sweep

Test different temperatures:

```bash
# Low temperature (conservative)
--temperature 0.3
# Should be: very predictable, less creative

# Medium temperature (balanced)
--temperature 0.8
# Should be: creative but coherent

# High temperature (creative)
--temperature 1.5
# Should be: very creative, may be nonsensical
```

### Prompt Sensitivity

Test various prompts:

```bash
# Simple
--prompt "The"

# Complex
--prompt "In the year 2050, scientists discovered"

# Rare words
--prompt "Quantum entanglement"

# Different styles
--prompt "Dear Sir or Madam,"  # Formal
--prompt "hey whats up"         # Casual
```

**Good model:** Adapts to prompt style and content.

## Benchmarks

### Standard Benchmarks

For character-level models:

| Dataset | Description | Good Perplexity |
|---------|-------------|-----------------|
| Penn Treebank | News text | <2.0 |
| Text8 | Wikipedia | <1.5 |
| Enwik8 | Wikipedia + markup | <1.3 |
| Shakespeare | Plays/sonnets | <1.8 |

### Custom Benchmarks

Create domain-specific tests:

```bash
# Medical text
fundamentallm evaluate model.pt --data medical_test.txt

# Code
fundamentallm evaluate model.pt --data code_test.txt

# Dialogue
fundamentallm evaluate model.pt --data dialogue_test.txt
```

## Comparative Evaluation

### Model Comparison

```bash
# Train different models
fundamentallm train data.txt --model-dim 128 --output-dir model_small
fundamentallm train data.txt --model-dim 256 --output-dir model_medium
fundamentallm train data.txt --model-dim 512 --output-dir model_large

# Compare
for model in model_*/final.pt; do
    echo "Evaluating $model"
    fundamentallm evaluate $model --data test.txt
done
```

**Create comparison table:**

| Model | Size | Perplexity | Time |
|-------|------|------------|------|
| Small | 128d | 8.4 | 5 min |
| Medium | 256d | 5.2 | 15 min |
| Large | 512d | 3.8 | 45 min |

### A/B Testing

Compare specific changes:

```bash
# Baseline
fundamentallm train data.txt --output-dir baseline

# With dropout
fundamentallm train data.txt --dropout 0.2 --output-dir with_dropout

# Compare
fundamentallm evaluate baseline/final.pt --data test.txt
fundamentallm evaluate with_dropout/final.pt --data test.txt
```

## Advanced Metrics

### Token-level Accuracy

**What it measures:** How often model predicts correct next token.

```python
def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    
    for batch in data_loader:
        predictions = model(batch['input'])
        predicted_tokens = predictions.argmax(dim=-1)
        
        correct += (predicted_tokens == batch['target']).sum()
        total += batch['target'].numel()
    
    return correct / total
```

**Interpretation:**
- Random guess (256 chars): 0.39%
- Weak model: 10-30%
- Good model: 50-70%
- Strong model: 70-85%

### Calibration

**What it measures:** Are probabilities well-calibrated?

```python
# If model says 80% confident, is it right 80% of the time?
```

**Why it matters:** 
- Confidence estimates
- Uncertainty quantification
- Beam search effectiveness

### Diversity Metrics

**Measure output variety:**

```python
def diversity(samples):
    # Unique n-grams / total n-grams
    n = 4  # 4-grams
    ngrams = set()
    total = 0
    
    for sample in samples:
        for i in range(len(sample) - n + 1):
            ngrams.add(sample[i:i+n])
            total += 1
    
    return len(ngrams) / total
```

**Interpretation:**
- Low diversity: Repetitive, boring
- High diversity: Creative, varied

## Error Analysis

### Common Mistakes

**1. Spelling Errors**
```
Model output: "definately" (should be "definitely")
```

**2. Grammar Errors**
```
Model output: "They was happy"
```

**3. Repetition**
```
Model output: "the the the the"
```

**4. Inconsistency**
```
Start: "John went to the store."
Later: "Sarah bought milk."  (who?)
```

### Categorizing Errors

```python
def analyze_errors(model, test_data):
    errors = {
        'spelling': [],
        'grammar': [],
        'repetition': [],
        'coherence': [],
    }
    
    for sample in test_data:
        output = model.generate(sample['prompt'])
        
        # Check for issues
        if has_spelling_error(output):
            errors['spelling'].append(output)
        # ... etc
    
    return errors
```

## Evaluation Pipeline

### Complete Workflow

```bash
#!/bin/bash
# evaluate.sh - Complete evaluation pipeline

MODEL=$1
echo "Evaluating $MODEL"

# 1. Quantitative metrics
echo "=== Test Set Evaluation ==="
fundamentallm evaluate $MODEL --data test.txt

# 2. Sample generation
echo "=== Sample Generation ==="
fundamentallm generate $MODEL \ \
    --prompt "Once upon a time" \ \
    --num-samples 3 \ \
    --temperature 0.8

# 3. Temperature sweep
echo "=== Temperature Sweep ==="
for temp in 0.3 0.8 1.2; do
    echo "Temperature: $temp"
    fundamentallm generate $MODEL \ \
    --prompt "The quick" \ \
    --temperature $temp \ \
    --max-length 50
done

# 4. Prompt diversity
echo "=== Prompt Diversity ==="
for prompt in "Hello" "In the year" "Dear Sir"; do
    echo "Prompt: $prompt"
    fundamentallm generate $MODEL \ \
    --prompt "$prompt" \ \
    --max-length 50
done
```

Usage:
```bash
bash evaluate.sh model.pt
```

## Continuous Evaluation

### During Training

Monitor metrics in real-time:

```bash
# Training with logging
fundamentallm train data.txt \ \
    --validation-data val.txt \ \
    --log-interval 100 \ \
    --eval-interval 1000

# Outputs:
# Step 100: loss=3.45
# Step 200: loss=3.12
# Step 1000: val_loss=2.87, perplexity=17.6
```

### Checkpointing

Evaluate each checkpoint:

```bash
# Save regular checkpoints
--save-interval 5000

# Evaluate all
for checkpoint in checkpoints/step_*.pt; do
    fundamentallm evaluate $checkpoint --data val.txt
done
```

Find best checkpoint (not necessarily the last!).

## Practical Tips

### 1. Multiple Seeds

Run with different random seeds:

```bash
for seed in 1 2 3 4 5; do
    fundamentallm train data.txt \ \
    --seed $seed \ \
    --output-dir run_$seed
done

# Average results for robustness
```

### 2. Statistical Significance

Use multiple runs to compute confidence intervals:

```python
import numpy as np

perplexities = [4.2, 4.5, 4.1, 4.3, 4.4]
mean = np.mean(perplexities)
std = np.std(perplexities)

print(f"Perplexity: {mean:.2f} ± {std:.2f}")
```

### 3. Failure Cases

Collect and analyze failure cases:

```bash
# Find worst predictions
# Generate samples that are nonsensical
# Understand what model struggles with
```

### 4. Human Evaluation

**Sometimes necessary!**

```
Ask humans to rate:
1. Fluency (1-5)
2. Coherence (1-5)
3. Relevance to prompt (1-5)
4. Overall quality (1-5)
```

## Visualization

### Loss Curves

```python
import matplotlib.pyplot as plt
import json

# Load training logs
with open('train_log.json') as f:
    log = json.load(f)

plt.figure(figsize=(12, 4))

# Training loss
plt.subplot(1, 2, 1)
plt.plot(log['steps'], log['train_loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Validation metrics
plt.subplot(1, 2, 2)
plt.plot(log['eval_steps'], log['val_loss'], label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Attention Visualization

Visualize what model attends to:

```python
# Get attention weights
attention = model.get_attention_weights(input_text)

# Plot heatmap
plt.imshow(attention, cmap='viridis')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar()
```

## Regression Testing

**Ensure new changes don't hurt performance:**

```bash
# 1. Establish baseline
fundamentallm evaluate baseline_model.pt --data test.txt > baseline.txt

# 2. After changes
fundamentallm evaluate new_model.pt --data test.txt > new.txt

# 3. Compare
diff baseline.txt new.txt
```

## Further Reading

- [Training Guide](./training.md) - Training process
- [Hyperparameters](./hyperparameters.md) - Tuning for better metrics
- [Generation](./generation.md) - Sampling strategies
- "Perplexity" (Wikipedia) - Understanding the metric
- BPE paper (Sennrich et al., 2016) - For subword models

## Next Steps

- [Training Deep-Dive](../tutorials/training-deep-dive.md) - Optimize training
- [Advanced Generation](../tutorials/advanced-generation.md) - Better sampling
- [Troubleshooting](./troubleshooting.md) - Fix issues
