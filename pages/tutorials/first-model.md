# Tutorial: Your First Model

Train and use your first transformer language model in 10 minutes.

## What You'll Learn

- How to prepare data for training
- How to train a small transformer model
- How to generate text from your model
- How to evaluate model quality

**Time required:** 10-15 minutes

## Prerequisites

- FundamentaLLM installed ([Installation Guide](./installation.md))
- Python virtual environment activated
- ~500MB free disk space

## Step 1: Understand the Data

Let's look at what we're training on:

```bash
# View sample data
cat data/samples/sample_data.txt | head -20
```

This is a small text file. Language models learn from any text - books, code, dialogue, anything.

**Character-level insight:** The model will learn patterns like:
- "Th" is often followed by "e"
- After "." comes a space and capital letter
- Common words like "the", "and", "is"

## Step 2: Train a Tiny Model (2 minutes)

Let's start with a very small, fast-training model:

```bash
fundamentallm train data/samples/sample_data.txt \ \
    --output-dir my_first_model \ \
    --model-dim 64 \ \
    --num-heads 2 \ \
    --num-layers 2 \ \
    --epochs 5 \ \
    --batch-size 16
```

**What each parameter means:**
- `--model-dim 64`: Small hidden size (faster)
- `--num-heads 2`: Two attention heads
- `--num-layers 2`: Only two transformer layers
- `--epochs 5`: Five passes through data
- `--batch-size 16`: Process 16 sequences at once

**Expected output:**
```
╭─── Training Configuration ───╮
│ Model: Transformer           │
│ Parameters: 156,234          │
│ Hidden dim: 64               │
│ Layers: 2                    │
│ Attention heads: 2           │
╰──────────────────────────────╯

Epoch 1/5
████████████████████ 100% | Loss: 4.234 | Val Loss: 4.156
Checkpoint saved: my_first_model/checkpoint_epoch_1.pt

Epoch 2/5
████████████████████ 100% | Loss: 3.891 | Val Loss: 3.823
...
```

Training should take ~2 minutes on a CPU, <30 seconds on GPU.

**What's happening:**
1. Text is converted to character IDs
2. Model learns to predict next character
3. Loss decreases = model improves
4. Checkpoints saved after each epoch

## Step 3: Inspect Training Outputs

```bash
# List what was created
ls -lh my_first_model/
```

You should see:
```
checkpoint_epoch_1.pt    # After epoch 1
checkpoint_epoch_2.pt    # After epoch 2
...
checkpoint_epoch_5.pt    # After epoch 5
final_model.pt           # Best model
training_log.txt         # Metrics log
config.yaml              # Hyperparameters used
```

**View training log:**
```bash
cat my_first_model/training_log.txt
```

Look for:
- Decreasing loss (good!)
- Final perplexity value
- Training time per epoch

## Step 4: Generate Text

Now let's use the trained model:

```bash
fundamentallm generate my_first_model/final_model.pt \ \
    --prompt "The" \ \
    --max-tokens 50 \ \
    --temperature 0.8
```

**Expected output:**
```
The ancient scrolls spoke of a time when...
```

The text won't be perfect (we only trained for 5 epochs on small data), but should be somewhat coherent!

**Try different prompts:**
```bash
# Different starting points
--prompt "Once upon a time"
--prompt "In the beginning"
--prompt "Chapter 1:"
```

**Try different temperatures:**
```bash
# Focused (predictable)
--temperature 0.3

# Balanced
--temperature 0.8

# Creative (random)
--temperature 1.5
```

## Step 5: Interactive Exploration

The most fun way to play with your model:

```bash
fundamentallm generate my_first_model/final_model.pt --interactive
```

**Try this:**
```
> The cat
The cat walked slowly down the ancient corridor...

> /set temperature=0.5
Updated: temperature=0.5

> The cat
The cat sat quietly on the windowsill.

> /set temperature=1.5
Updated: temperature=1.5

> The cat  
The cat danced beneath forgotten starlight dreams...

> /quit
```

**Observation:** Higher temperature = more creative but less coherent.

## Step 6: Evaluate the Model

```bash
fundamentallm evaluate my_first_model/final_model.pt \
    data/samples/sample_data.txt
```

**Output:**
```
╭─── Model Evaluation ───╮
│ Test Loss:   3.234     │
│ Perplexity:  25.3      │
│ Bits/Char:   3.65      │
╰────────────────────────╯
```

**Interpreting metrics:**
- **Perplexity:** How "surprised" the model is. Lower = better.
  - Random guessing: ~256 (vocab size)
  - Decent model: 10-30
  - Good model: 3-10
  - Great model: 1.5-3
  
- **Bits per character:** Information content. Lower = better compression.

## Step 7: Train a Better Model

Now let's train a larger, better model:

```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \ \
    --output-dir shakespeare_model \ \
    --model-dim 256 \ \
    --num-heads 8 \ \
    --num-layers 6 \ \
    --epochs 20 \ \
    --batch-size 32
```

**Differences from first model:**
- More data (shakespeare100k.txt)
- Larger model (256 vs 64 hidden dim)
- More attention heads (8 vs 2)
- Deeper (6 vs 2 layers)
- More training (20 vs 5 epochs)

**This will take ~15-30 minutes depending on your hardware.**

While it trains, let's understand what's happening...

## Understanding Training

### The Learning Process

```
Step 1: Model sees "To be or not to b"
Step 2: Predicts next character
Step 3: Actual next character is "e"
Step 4: Compare prediction to truth → compute loss
Step 5: Adjust model weights to improve
Step 6: Repeat for millions of character sequences
```

### Watching Progress

Training prints:
```
Epoch 3/20
████████████████████ 45% | Loss: 2.134 | ETA: 3m12s
```

**Good signs:**
- Loss decreasing steadily
- No NaN or extremely high values
- Perplexity dropping

**Bad signs:**
- Loss not changing
- Loss = NaN (learning rate too high)
- Validation loss increasing (overfitting)

## Step 8: Compare Models

Once the Shakespeare model finishes:

```bash
# Generate from both models
echo "=== Tiny Model ===" fundamentallm generate my_first_model/final_model.pt \ \
    --prompt "To be or not to be" \ \
    --max-tokens 100

echo "\n=== Shakespeare Model ==="
fundamentallm generate shakespeare_model/final_model.pt \ \
    --prompt "To be or not to be" \ \
    --max-tokens 100
```

The Shakespeare model should produce much more coherent, Shakespeare-like text!

## Step 9: Experiment

Try these experiments:

### Experiment 1: Different Data

```bash
# Train on your own text file
fundamentallm train my_text.txt \ \
    --output-dir my_custom_model \ \
    --epochs 20
```

The model will learn the style of your data!

### Experiment 2: Hyperparameters

```bash
# Very small model (fast)
--model-dim 32 --num-layers 2

# Medium model (balanced)
--model-dim 256 --num-layers 6

# Large model (slow but powerful)
--model-dim 512 --num-layers 12
```

### Experiment 3: Temperature

```bash
# Conservative
--temperature 0.3

# Standard
--temperature 0.8

# Wild
--temperature 2.0
```

## What You've Learned

✅ How to train a transformer from scratch  
✅ How character-level tokenization works  
✅ What training metrics mean  
✅ How to generate text with sampling  
✅ Effect of model size and training time  
✅ How temperature controls creativity  

## Common Issues & Solutions

### Issue: "Out of memory"

**Solution:** Reduce batch size or model size
```bash
--batch-size 8 --model-dim 128
```

### Issue: Loss not decreasing

**Solution:** Increase learning rate or model size
```bash
--learning-rate 0.01 --model-dim 512
```

### Issue: Generated text is nonsense

**Solution:** Train longer or on more data
```bash
--epochs 50
```

### Issue: Training is slow

**Solution:** Use GPU or reduce model size
```bash
--device cuda  # If you have GPU
# Or
--model-dim 128 --num-layers 4  # Smaller model
```

## Next Steps

Now that you've trained your first model, explore:

### Understand the Theory
- [Transformer Architecture](../concepts/transformers.md) - How it works
- [Attention Mechanism](../concepts/attention.md) - The key innovation
- [Language Modeling](../concepts/language-modeling.md) - The learning objective

### Improve Your Models
- [Training Guide](../guide/training.md) - Optimize hyperparameters
- [Hyperparameter Tuning](../guide/hyperparameters.md) - What each setting does
- [Data Preparation](../guide/data-prep.md) - Better data = better models

### Go Deeper
- [Training Deep Dive](./training-deep-dive.md) - Detailed walkthrough
- [Advanced Generation](./advanced-generation.md) - Control generation better
- [Custom Datasets](./custom-datasets.md) - Use your own data

### Explore Implementation
- [Models Module](../modules/models.md) - How transformers are built
- [Training Module](../modules/training.md) - Training loop internals
- [Generation Module](../modules/generation.md) - Sampling strategies

## Congratulations!

You've successfully:
- 🎉 Trained your first transformer model
- 🎉 Generated text from it
- 🎉 Understood the training process
- 🎉 Experimented with parameters

You're ready to explore deeper topics in language modeling!

## Quick Reference

### Train tiny model (2 min)
```bash
fundamentallm train data/samples/sample_data.txt \ \
    --output-dir tiny --epochs 5
```

### Train good model (30 min)
```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \ \
    --output-dir good --epochs 20 --model-dim 256
```

### Generate
```bash
fundamentallm generate model.pt --prompt "Text" --temperature 0.8
```

### Interactive
```bash
fundamentallm generate model.pt --interactive
```

### Evaluate
```bash
fundamentallm evaluate model.pt data.txt
```
