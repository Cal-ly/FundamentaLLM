# Training Models

Comprehensive guide to training transformer language models with FundamentaLLM.

## Quick Start

```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \
    --output-dir my_model \
    --epochs 20 \
    --batch-size 32
```

This trains a model on Shakespeare's works. Let's understand what happens and how to optimize it.

## Training Pipeline

### The Complete Flow

```
Data → Tokenizer → Batches → Model → Loss → Gradients → Optimizer → Updated Model
  ↑                                                                         ↓
  └─────────────────────────────────────────────────────────────────────────┘
                              Repeat for N epochs
```

### Step-by-Step

**1. Data Loading**
```python
# Load text file
text = open('data.txt').read()

# Tokenize (character-level)
tokens = tokenizer.encode(text)  # [72, 101, 108, ...]
```

**2. Create Sequences**
```python
# Split into training sequences
sequences = create_sequences(tokens, max_len=256)
# Each sequence: [batch_size, seq_len]
```

**3. Forward Pass**
```python
# Run through model
logits = model(sequences)  # [batch, seq_len, vocab_size]

# Shift for next-token prediction
predictions = logits[:, :-1]
targets = sequences[:, 1:]
```

**4. Compute Loss**
```python
# Cross-entropy loss
loss = F.cross_entropy(predictions.view(-1, vocab_size),
                       targets.view(-1))
```

**5. Backward Pass**
```python
# Compute gradients
loss.backward()

# Clip gradients (prevent explosion)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**6. Update Weights**
```python
# Optimizer step
optimizer.step()
optimizer.zero_grad()
```

## Hyperparameters

### Model Architecture

#### Hidden Dimension (`--model-dim`)
**What:** Size of embedding and hidden layers  
**Default:** 128  
**Typical range:** 64-1024

```bash
# Small model (fast, less capacity)
--model-dim 64

# Medium model (balanced)
--model-dim 256

# Large model (slow, more capacity)
--model-dim 512
```

**Rule of thumb:**
- More data → bigger model
- Less compute → smaller model

#### Number of Layers (`--num-layers`)
**What:** Depth of the transformer  
**Default:** 6  
**Typical range:** 3-24

```bash
# Shallow (fast)
--num-layers 3

# Medium (standard)
--num-layers 6

# Deep (slow but powerful)
--num-layers 12
```

**Trade-off:** More layers = better representations but slower training and risk of overfitting.

#### Attention Heads (`--num-heads`)
**What:** Parallel attention mechanisms  
**Default:** 2  
**Typical range:** 2-16

```bash
# Few heads (simpler)
--num-heads 2

# Standard
--num-heads 8

# Many heads (more complex patterns)
--num-heads 16
```

**Constraint:** Must divide `model_dim` evenly.
```
model_dim=256, num_heads=8 → head_dim=32 ✓
model_dim=256, num_heads=7 → ERROR ✗
```

#### Feedforward Dimension (`--ff-dim`)
**What:** Size of feedforward network inside transformer  
**Default:** `4 × model_dim`  
**Typical multiplier:** 4x

Standard practice: FFN is 4× larger than hidden dimension.

### Training Configuration

#### Learning Rate (`--learning-rate`)
**What:** How much to update weights each step  
**Default:** 0.001 (1e-3)  
**Typical range:** 1e-5 to 1e-2

```bash
# Conservative (safe but slow)
--learning-rate 0.0001

# Standard (good starting point)
--learning-rate 0.001

# Aggressive (fast but risky)
--learning-rate 0.01
```

**Signs of wrong learning rate:**
- Too high: Loss explodes or NaN
- Too low: Loss barely decreases

#### Batch Size (`--batch-size`)
**What:** Number of sequences processed in parallel  
**Default:** 16  
**Typical range:** 8-128

```bash
# Small (less memory, noisier gradients)
--batch-size 8

# Medium (balanced)
--batch-size 32

# Large (more memory, stable gradients)
--batch-size 64
```

**Memory requirement:** Proportional to `batch_size × seq_len`

**Trade-off:**
- Larger batches: More stable, better GPU utilization, more memory
- Smaller batches: Less memory, faster iteration, noisier gradients

#### Epochs (`--epochs`)
**What:** Complete passes through the dataset  
**Default:** 10  
**Typical range:** 5-100

```bash
# Quick test
--epochs 5

# Standard training
--epochs 20

# Long training (risk overfitting)
--epochs 50
```

**Watch for:** Validation loss increasing while training loss decreases = overfitting

#### Dropout (`--dropout`)
**What:** Regularization by randomly dropping connections  
**Default:** 0.1  
**Typical range:** 0.0-0.3

```bash
# No regularization
--dropout 0.0

# Light regularization
--dropout 0.1

# Heavy regularization (prevent overfitting)
--dropout 0.3
```

Higher dropout when:
- Small dataset
- Signs of overfitting
- Large model

### Sequence Configuration

#### Max Sequence Length (`--max-seq-len`)
**What:** Maximum tokens per training sequence  
**Default:** 256  
**Typical range:** 128-2048

```bash
# Short sequences (less memory)
--max-seq-len 128

# Standard
--max-seq-len 256

# Long sequences (more context, more memory)
--max-seq-len 512
```

**Memory:** Attention is O(n²), so doubling sequence length = 4× memory

#### Validation Split (`--val-split`)
**What:** Fraction of data for validation  
**Default:** 0.1 (10%)  
**Typical range:** 0.05-0.2

```bash
# Small validation set
--val-split 0.05

# Standard
--val-split 0.1

# Large validation set
--val-split 0.2
```

## Training Strategies

### 1. Start Small, Scale Up

```bash
# Step 1: Quick test (2 minutes)
fundamentallm train data/samples/sample_data.txt \
    --output-dir test_model \
    --model-dim 64 \
    --num-layers 2 \
    --epochs 3

# Step 2: Medium training (20 minutes)
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \
    --output-dir medium_model \
    --model-dim 256 \
    --num-layers 6 \
    --epochs 20

# Step 3: Full training (hours)
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt \
    --output-dir large_model \
    --model-dim 512 \
    --num-layers 12 \
    --epochs 50 \
    --batch-size 64
```

### 2. Monitor Training

During training, you'll see output like:
```
Epoch 1/20
[████████░░] 50%  |  train_loss=2.934 | val_loss=2.509 | lr=1.37e-04 | throughput=98091 tokens/sec
```

**Understanding Training Output**

#### Loss Metrics

**`train_loss`** - Average training loss across batches
- **What it means:** How well the model predicts the next token during training
- **Conceptually:** Lower = model is learning the training patterns better
- **Ideal behavior:** Smooth, steady decrease over epochs
- **Watch for:**
  - Erratic spikes → Learning rate too high (exploding gradients)
  - Completely flat → Learning rate too low or model converged
  - Slight noise → Normal, indicates stochastic gradient descent working

**Example interpretation:**
```
Epoch 1: train_loss=3.215  ← Model is initially guessing poorly
Epoch 5: train_loss=2.120  ← Getting better at predictions
Epoch 10: train_loss=1.804 ← Fine-tuning the learned patterns
```

**`val_loss`** - Loss measured on validation data (unseen during training)
- **What it means:** How well the model generalizes to new data
- **Conceptually:** The real test of model performance
- **Ideal behavior:** Decreases along with training loss, following similar trend
- **Watch for:**
  - **Diverging:** val_loss increases while train_loss decreases = **overfitting**
    - Model is memorizing training data instead of learning generalizable patterns
    - Solution: reduce `--dropout`, use less epochs, or get more training data
  - **Both increasing:** Configuration issue (bad learning rate, bad data)
  - **Both flat:** Model converged or hit a learning plateau

**Key insight - Why both matter:**
```
Good training (Epoch 10):
  train_loss=1.5, val_loss=1.7  ← Similar, good generalization ✅

Overfitting (Epoch 15):
  train_loss=0.8, val_loss=2.5  ← Huge gap, memorizing ❌

Underfitting (Epoch 10):
  train_loss=3.0, val_loss=3.1  ← Both high, needs more capacity ❌
```

#### Learning & Speed Metrics

**`lr` (Learning Rate)**
- Current learning rate at this step
- **Conceptually:** How big are the weight updates?
- **Changes over time:** If using `--lr-schedule`, this will decay
- **Example:** `lr=1.37e-04` means updating weights by tiny fractions

**`throughput` (tokens/sec)**
- How many tokens per second your model processes
- **Affects total training time:** More tokens/sec = faster training
- **Factors that reduce throughput:**
  - Larger model (more computation)
  - Larger batch size initially speeds up, then hits memory limits
  - GPU memory pressure

#### Perplexity (if displayed)

- **What it is:** $\text{Perplexity} = e^{\text{loss}}$
- **Conceptually:** "How many equally-likely choices does the model think there are?"
- **Intuition:**
  - Perplexity = 1 → Perfect predictions
  - Perplexity = 100 → Model thinks 100 tokens are equally likely
  - Lower is better
- **Target ranges:**
  - Character-level: 1.5-3.0 (reasonable)
  - Token-level: 10-50 (reasonable)

#### Validation Loss Patterns

**Pattern: Both losses decreasing together** ✅ Good
```
Epoch  train_loss  val_loss
1      3.200       3.180
5      2.100       2.090
10     1.500       1.520
```
Model is learning generalizable patterns. Continue training or stop when loss plateaus.

**Pattern: Train decreasing, val increasing** ❌ Overfitting
```
Epoch  train_loss  val_loss
1      3.200       3.180
5      1.800       2.200
10     0.900       2.800
```
Model is memorizing training data. Fix by:
- Reducing epochs
- Increasing dropout (`--dropout 0.2`)
- Using less model capacity (`--model-dim 128` instead of 256)

**Pattern: Both increasing or flat** ❌ Configuration issue
```
Epoch  train_loss  val_loss
1      3.200       3.180
5      3.100       3.220
10     3.050       3.250
```
Possible causes:
- Learning rate too high (gradients exploding)
- Learning rate too low (barely learning)
- Bad data or data pipeline issue
- Model too small for the task

### Best Practices for Monitoring

1. **Watch the first epoch closely**
   - First few steps: loss should rapidly decrease (exponentially)
   - If loss doesn't decrease: learning rate too low

2. **Compare train vs validation**
   - Gap of 5-15% is normal and healthy
   - Gap > 50% suggests overfitting

3. **Check every N epochs, not every step**
   - Focus on overall trend, not noise
   - Small fluctuations are normal

4. **Save checkpoints at best validation loss**
   - Model automatically saves when val_loss improves
   - This prevents overfitting by using the best-performing version

### 3. Learning Rate Scheduling

Watch these metrics:

**Training Loss**
- Should decrease steadily
- Erratic = learning rate too high
- Flat = learning rate too low or converged

**Validation Loss**
- Should decrease (following training loss)
- Increasing while training decreases = overfitting

**Perplexity**
- Lower is better
- Character-level: aim for 1.5-3.0

**Generation Quality**
- Periodically generate text to check
- Qualitative assessment of coherence

Use learning rate schedules for better convergence:

```bash
# Warm-up then decay
--lr-schedule cosine \
--warmup-steps 1000
```

**Common schedules:**
- **Constant:** Same LR throughout
- **Step decay:** Reduce LR at intervals
- **Cosine:** Smooth decay following cosine curve
- **Warmup:** Start low, ramp up, then decay

### 4. Checkpointing & Model Selection

FundamentaLLM automatically saves multiple versions during training:

```
my_model/
├── checkpoint_epoch_1.pt      # Checkpoint after epoch 1
├── checkpoint_epoch_2.pt      # Checkpoint after epoch 2
...
├── best.pt                     # Model with lowest validation loss
└── final_model.pt             # Model after training completed
```

#### Understanding Best vs Final Models

**`best.pt`** - The model that generalized best during training
- **When to use:** Usually the preferred choice for generation/deployment
- **Why it exists:** Captures the point where validation loss was lowest
- **Conceptually:** This is the "sweet spot" before overfitting became severe
- **Created when:** FundamentaLLM finds a new best validation loss

**`final_model.pt`** - The model after all epochs completed
- **When to use:** Only if you specifically want the latest state
- **Caution:** May be overfit if validation loss was increasing near the end
- **Example case:**
  ```
  Epoch 10: val_loss=1.50 ← best.pt saved here (best generalization)
  Epoch 15: val_loss=1.55 ← val_loss increasing (overfitting)
  Epoch 20: val_loss=1.63 ← final_model.pt here (worst generalization)
  ```

#### Which Model to Use?

**Default recommendation:** Use `best.pt` for generation and deployment
```bash
# Best practice - use the model with best validation performance
fundamentallm generate my_model/best.pt --prompt "Once upon a time"
```

**When final_model.pt might be better:**
- If you cut training short and validation loss was still decreasing
- If you used very aggressive regularization and overfitting wasn't an issue
- If you want the latest state for resuming training

#### Why They Can Differ

Training on the same data with same parameters, `best.pt` and `final_model.pt` differ because:

1. **Training trajectory matters**
   - Early epochs: Model is learning patterns
   - Middle epochs: Model improves on validation data
   - Late epochs: Model overfits, validation performance degrades

2. **Overfitting dynamics**
   - Model continues learning training patterns even after generalizing best
   - Eventually starts memorizing training-specific details
   - Validation loss increases when memorization overtakes learning

3. **Optimal stopping point**
   - Best validation loss = best general learning point
   - Final loss = after model had time to memorize

#### Example: Monitoring Which is Better

```
Model Training Progress:
Epoch  train_loss  val_loss   Selected
1      3.200       3.180      
2      2.800       2.750      
3      2.500       2.400      
4      2.200       2.150      
5      1.900       2.000      best.pt ⭐ (validation stops improving)
6      1.600       2.050      
7      1.300       2.180      
8      1.100       2.350      
9      0.900       2.520      
10     0.750       2.680      final_model.pt (overfitting evident)
```

At epoch 5, the model has learned to generalize best. After that, it's memorizing training details, causing validation loss to increase.

#### Manual Checkpoint Resumption

If you want to resume training from `best.pt`:
```bash
fundamentallm train data.txt \
    --output-dir my_model \
    --resume-from my_model/best.pt \
    --epochs 30  # Continue for more epochs
```

### 5. Mixed Precision Training

Speed up training with half-precision floats:

```bash
--mixed-precision
```

**Benefits:**
- 2-3× faster training
- Less memory usage
- Same model quality

**Requirements:**
- NVIDIA GPU with Tensor Cores
- PyTorch AMP support

## Example Configurations

### Tiny Model (Testing)

```bash
fundamentallm train data/samples/sample_data.txt \
    --output-dir tiny_model \
    --model-dim 64 \
    --num-heads 2 \
    --num-layers 2 \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 0.001
```
Time: ~2 minutes  
Use: Pipeline testing

### Small Model (Learning)

```bash
fundamentallm train data/raw/shakespeare/shakespeare25k.txt \
    --output-dir small_model \
    --model-dim 128 \
    --num-heads 4 \
    --num-layers 4 \
    --batch-size 32 \
    --epochs 20 \
    --learning-rate 0.001
```
Time: ~15 minutes  
Use: Understanding training dynamics

### Medium Model (Decent Quality)

```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \
    --output-dir medium_model \
    --model-dim 256 \
    --num-heads 8 \
    --num-layers 6 \
    --batch-size 32 \
    --epochs 30 \
    --learning-rate 0.0005 \
    --dropout 0.1 \
    --mixed-precision
```
Time: ~1 hour  
Use: Quality text generation

### Large Model (Best Quality)

```bash
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt \
    --output-dir large_model \
    --model-dim 512 \
    --num-heads 16 \
    --num-layers 12 \
    --batch-size 64 \
    --epochs 50 \
    --learning-rate 0.0003 \
    --dropout 0.2 \
    --mixed-precision \
    --gradient-clip 1.0
```
Time: ~8 hours  
Use: Production quality

## Common Issues

### Loss is NaN

**Cause:** Learning rate too high or gradient explosion  
**Fix:**
```bash
--learning-rate 0.0001  # Lower LR
--gradient-clip 1.0     # Clip gradients
```

### Out of Memory

**Cause:** Batch size or sequence length too large  
**Fix:**
```bash
--batch-size 16         # Reduce batch size
--max-seq-len 128       # Shorter sequences
--model-dim 256         # Smaller model
```

### Loss Not Decreasing

**Cause:** Learning rate too low, or model too small  
**Fix:**
```bash
--learning-rate 0.01    # Increase LR
--model-dim 512         # Larger model
--num-layers 8          # Deeper model
```

### Overfitting

**Symptoms:** Validation loss increases while training loss decreases  
**Fix:**
```bash
--dropout 0.2           # More regularization
--epochs 15             # Train less
# Or: Get more data
```

## Advanced Techniques

### Gradient Accumulation

Simulate larger batches:

```bash
--batch-size 8 \
--accumulation-steps 4  # Effective batch size: 32
```

### Early Stopping

Stop when validation loss stops improving:

```bash
--early-stopping \
--patience 5  # Stop if no improvement for 5 epochs
```

### Data Augmentation

For character-level models:
- Random casing changes
- Punctuation variations
- Whitespace normalization

## Monitoring Tools

### TensorBoard

```bash
# Log to TensorBoard
--tensorboard --log-dir runs/

# View logs
tensorboard --logdir runs/
```

### W&B Integration

```bash
# Track with Weights & Biases
--wandb --project fundamentallm
```

## Next Steps

- [Hyperparameter Tuning](./hyperparameters.md) - Optimize settings
- [Evaluation](./evaluation.md) - Measure quality
- [Generation](./generation.md) - Use trained models
- [Training Module](../modules/training.md) - Implementation details
