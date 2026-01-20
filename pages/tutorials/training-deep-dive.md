# Training Deep-Dive

Advanced training techniques and best practices for optimal model performance.

## Complete Training Workflow

### 1. Data Preparation

```bash
# Check data quality
python scripts/analyze_data.py data/raw/my_text.txt

# Clean if needed
python scripts/clean_data.py data/raw/my_text.txt data/processed/clean.txt

# Split train/val
head -n 90000 data/processed/clean.txt > data/processed/train.txt
tail -n 10000 data/processed/clean.txt > data/processed/val.txt
```

### 2. Initial Experiment

```bash
# Quick baseline with small model
fundamentallm train data/processed/train.txt \
    --validation-data data/processed/val.txt \
    --output-dir experiments/baseline \
    --model-dim 128 \
    --num-layers 4 \
    --epochs 10 \
    --batch-size 32

# Check results
fundamentallm evaluate experiments/baseline/final.pt \
    --data data/processed/val.txt
```

### 3. Hyperparameter Search

```bash
# Grid search
for lr in 0.0001 0.001 0.01; do
    for dim in 128 256 512; do
        fundamentallm train data/processed/train.txt \
            --validation-data data/processed/val.txt \
            --output-dir experiments/lr_${lr}_dim_${dim} \
            --learning-rate $lr \
            --model-dim $dim \
            --epochs 20 \
            --early-stopping
    done
done

# Compare results
python scripts/compare_experiments.py experiments/
```

### 4. Final Training

```bash
# Best hyperparameters, longer training
fundamentallm train data/processed/train.txt \
    --validation-data data/processed/val.txt \
    --output-dir models/production \
    --learning-rate 0.001 \
    --model-dim 512 \
    --num-layers 8 \
    --num-heads 8 \
    --dropout 0.1 \
    --epochs 50 \
    --batch-size 64 \
    --lr-schedule cosine \
    --warmup-steps 2000 \
    --gradient-clip 1.0 \
    --mixed-precision \
    --early-stopping \
    --patience 10
```

## Advanced Techniques

### Learning Rate Warmup

**Implementation:**

```python
def get_lr(step, warmup_steps, max_lr, total_steps):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

# Usage
for step, batch in enumerate(data_loader):
    lr = get_lr(step, warmup_steps=1000, max_lr=0.001, total_steps=50000)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Why it works:**
- Prevents early instability
- Allows higher learning rates later
- Smoother convergence

### Mixed Precision Training

**Setup:**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in data_loader:
    optimizer.zero_grad()
    
    with autocast():
        # Forward in float16
        logits = model(batch['input'])
        loss = F.cross_entropy(logits, batch['target'])
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    
    # Gradient clipping (unscale first)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- **2x faster** training
- **50% less** memory
- Maintains accuracy (usually)

**When to use:**
- GPU with Tensor Cores (V100, A100, RTX 20/30/40 series)
- Memory-constrained

### Gradient Accumulation

**Implementation:**

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(data_loader):
    # Forward pass
    logits = model(batch['input'])
    loss = F.cross_entropy(logits, batch['target'])
    
    # Scale loss
    loss = loss / accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Step every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
```

**Use case:** Simulate large batch size with limited memory.

```
Actual batch: 16
Accumulation: 4
Effective batch: 64
```

## Monitoring and Logging

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/experiment_1')

for epoch in range(num_epochs):
    # Training
    train_loss = train_epoch(model, train_loader, optimizer)
    writer.add_scalar('Loss/train', train_loss, epoch)
    
    # Validation
    val_loss = validate(model, val_loader)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    # Learning rate
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    
    # Gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    # Weights
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)

writer.close()

# View in browser
# tensorboard --logdir runs/
```

### Weights & Biases (WandB)

```python
import wandb

# Initialize
wandb.init(
    project="fundamentallm",
    config={
        "learning_rate": 0.001,
        "architecture": "Transformer",
        "dataset": "shakespeare",
        "epochs": 20,
    }
)

# Log metrics
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })

# Log model
wandb.save('model.pt')
```

### Custom Logging

```python
import json
from pathlib import Path

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []
    
    def log(self, step, metrics):
        metrics['step'] = step
        self.metrics.append(metrics)
        
        # Save to file
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        steps = [m['step'] for m in self.metrics]
        train_loss = [m['train_loss'] for m in self.metrics]
        val_loss = [m['val_loss'] for m in self.metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss, label='Train')
        plt.plot(steps, val_loss, label='Validation')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.log_dir / 'loss_curve.png')

# Usage
logger = TrainingLogger('logs/experiment_1')

for step, batch in enumerate(train_loader):
    loss = train_step(model, batch, optimizer)
    
    if step % 100 == 0:
        val_loss = validate(model, val_loader)
        logger.log(step, {
            'train_loss': loss.item(),
            'val_loss': val_loss,
        })

logger.plot()
```

## Advanced Optimization

### Layer-wise Learning Rate Decay

**Idea:** Lower layers learn slower than higher layers.

```python
def get_layer_params(model):
    """Group parameters by layer."""
    groups = []
    
    # Embeddings
    groups.append({
        'params': model.embedding.parameters(),
        'lr': base_lr * 0.1,  # 10x slower
    })
    
    # Transformer layers
    for i, layer in enumerate(model.layers):
        decay = 0.9 ** (len(model.layers) - i - 1)
        groups.append({
            'params': layer.parameters(),
            'lr': base_lr * decay,
        })
    
    # Output layer
    groups.append({
        'params': model.output.parameters(),
        'lr': base_lr,
    })
    
    return groups

# Create optimizer with per-layer LR
optimizer = optim.AdamW(get_layer_params(model))
```

### Discriminative Fine-tuning

**For transfer learning:**

```python
# Freeze early layers
for param in model.embedding.parameters():
    param.requires_grad = False

for i, layer in enumerate(model.layers):
    if i < 4:  # Freeze first 4 layers
        for param in layer.parameters():
            param.requires_grad = False

# Train
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
```

### Lookahead Optimizer

**Wrapper for more stable training:**

```python
from torch_optimizer import Lookahead

# Base optimizer
base_optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lookahead wrapper
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

## Regularization Techniques

### Dropout Variants

**Standard dropout:**
```python
self.dropout = nn.Dropout(p=0.1)
```

**DropConnect (weight dropout):**
```python
# In forward pass
if self.training:
    mask = torch.bernoulli(torch.ones_like(self.weight) * (1 - self.dropconnect_p))
    weight = self.weight * mask
else:
    weight = self.weight
```

**Scheduled dropout:**
```python
# Increase dropout over training
current_dropout = min_dropout + (max_dropout - min_dropout) * (epoch / total_epochs)
model.set_dropout(current_dropout)
```

### Weight Decay Variants

**L2 regularization:**
```python
optimizer = optim.AdamW(model.parameters(), weight_decay=0.01)
```

**Exclude specific parameters:**
```python
# Don't apply weight decay to biases and LayerNorm
no_decay = ['bias', 'LayerNorm.weight']
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
    },
]
optimizer = optim.AdamW(param_groups, lr=0.001)
```

## Data Augmentation

### For Text

**Back-translation:**
```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src_lang='en', pivot_lang='fr'):
    # Translate to pivot language
    model_to = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}')
    tokenizer_to = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}')
    
    translated = model_to.generate(**tokenizer_to(text, return_tensors="pt"))
    pivot_text = tokenizer_to.decode(translated[0], skip_special_tokens=True)
    
    # Translate back
    model_back = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}')
    tokenizer_back = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}')
    
    back_translated = model_back.generate(**tokenizer_back(pivot_text, return_tensors="pt"))
    return tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
```

**Random insertion/deletion:**
```python
import random

def augment_text(text, p=0.1):
    chars = list(text)
    
    # Random deletion
    chars = [c for c in chars if random.random() > p]
    
    # Random insertion (duplicate)
    for i in range(len(chars)):
        if random.random() < p:
            chars.insert(i, chars[i])
    
    return ''.join(chars)
```

**For character-level, minimal augmentation usually sufficient.**

## Checkpointing Strategies

### Save Best Model

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Save if best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pt')
        print(f"Saved best model with val_loss={val_loss:.3f}")
```

### Save Multiple Checkpoints

```python
# Save every N epochs
if epoch % save_interval == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

# Keep only last K checkpoints
checkpoints = sorted(Path('checkpoints').glob('checkpoint_*.pt'))
if len(checkpoints) > max_checkpoints:
    for checkpoint in checkpoints[:-max_checkpoints]:
        checkpoint.unlink()
```

### Resume Training

```python
# Save complete state
torch.save({
    'epoch': epoch,
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'random_state': torch.get_rng_state(),
}, 'resume_checkpoint.pt')

# Resume
checkpoint = torch.load('resume_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
torch.set_rng_state(checkpoint['random_state'])
start_epoch = checkpoint['epoch'] + 1
```

## Debugging Training

### Check Gradients

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            ratio = grad_norm / (param_norm + 1e-7)
            
            print(f"{name:40s} | grad: {grad_norm:8.4f} | param: {param_norm:8.4f} | ratio: {ratio:.6f}")
            
            # Check for issues
            if torch.isnan(param.grad).any():
                print(f"  WARNING: NaN gradients in {name}")
            if torch.isinf(param.grad).any():
                print(f"  WARNING: Inf gradients in {name}")
            if grad_norm > 100:
                print(f"  WARNING: Large gradient in {name}")

# Use after backward, before optimizer step
loss.backward()
check_gradients(model)
optimizer.step()
```

### Gradient Flow Visualization

```python
def plot_grad_flow(named_parameters):
    """Plot gradient flow through network."""
    import matplotlib.pyplot as plt
    
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.5, label='Average')
    plt.bar(range(len(max_grads)), max_grads, alpha=0.5, label='Max')
    plt.hlines(0, 0, len(ave_grads), linestyle='--')
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel('Layers')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gradient_flow.png')

# Use after backward
loss.backward()
plot_grad_flow(model.named_parameters())
```

### Overfit Single Batch

**Test if model can learn:**

```python
# Get single batch
batch = next(iter(train_loader))

# Train on it
for step in range(1000):
    optimizer.zero_grad()
    logits = model(batch['input'])
    loss = F.cross_entropy(logits, batch['target'])
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

# Should reach near-zero loss
# If not, something is wrong with model/optimizer
```

## Further Reading

- [Hyperparameters Guide](../guide/hyperparameters.md) - Tuning
- [Troubleshooting](../guide/troubleshooting.md) - Common issues
- [Optimization](../concepts/optimization.md) - Theory
- "Practical recommendations for gradient-based training" (Bengio, 2012)

## Next Steps

- [Advanced Generation](./advanced-generation.md) - Better text generation
- [Custom Datasets](./custom-datasets.md) - Use your own data
- [Evaluation](../guide/evaluation.md) - Measure improvements
