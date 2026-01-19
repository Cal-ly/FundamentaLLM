# Phase 4: Training System - Handover Prompt

## Current State

**Completed:**
- ✅ Phase 1: Infrastructure (25+ files, configuration, base abstractions)
- ✅ Phase 2: Data Pipeline (tokenizer, dataset, dataloader, 18 tests)
- ✅ Phase 3: Model Architecture (Transformer with 5 components, 112 tests)
- **Total: 130 tests passing**

**Available Resources:**
- Complete Transformer model (`fundamentallm.models.transformer.Transformer`)
- Character-level tokenizer (`fundamentallm.data.tokenizer.CharacterTokenizer`)
- Language model dataset (`fundamentallm.data.dataset.LanguageModelDataset`)
- DataLoaders (train/val/test splits at token level)
- Config system (TransformerConfig, TrainingConfig via Pydantic + YAML)

---

## Phase 4 Objective

Build the **training system** to enable end-to-end model training on text datasets.

### Deliverables (6 Components)

#### **4.1 Loss Computation**
- **File:** `src/fundamentallm/training/losses.py`
- **Classes:**
  - `LanguageModelingLoss`: Cross-entropy loss with:
    - Shape handling: logits [batch, seq_len, vocab_size] → targets [batch, seq_len]
    - Reshape to [batch*seq_len, vocab_size] and [batch*seq_len] for efficient computation
    - Optional label smoothing (uniform distribution mixing)
    - Support for per-sample loss reduction
  - Utility: `compute_loss(logits, targets, reduction='mean')` function
- **Key Implementation Detail:** Ignore padding token loss (mask out -100 or -1 targets)
- **Tests:** Shape preservation, label smoothing, gradient flow (3+ tests)

#### **4.2 Optimizer Builder** 
- **File:** `src/fundamentallm/training/optimizers.py`
- **Classes:**
  - `OptimizerBuilder`: Factory for common optimizers
    - AdamW (default, betas=(0.9, 0.999), weight decay=0.01)
    - Adam (no weight decay)
    - SGD (momentum=0.9)
    - RMSprop (alpha=0.99)
  - Method: `build(optimizer_name, model, lr, **kwargs) → torch.optim.Optimizer`
  - Support parameter groups (e.g., no weight decay for bias/norm layers)
- **Key Implementation Detail:** Weight decay only on linear layers, not bias/LayerNorm
- **Tests:** All optimizer types, learning rate application, parameter groups (4+ tests)

#### **4.3 Learning Rate Scheduler**
- **File:** `src/fundamentallm/training/schedulers.py`
- **Classes:**
  - `LearningRateScheduler`: Base class for scheduling strategies
  - Concrete implementations:
    - `ConstantLRScheduler`: Fixed LR (baseline)
    - `LinearWarmup`: LR ramps 0 → target over N steps, then constant
    - `CosineAnnealingScheduler`: Cosine decay from target → min_lr over total steps
    - `ExponentialDecayScheduler`: Exponential decay: `lr = initial_lr * decay_rate^epoch`
  - Method: `step() → lr` per batch/epoch
  - Integrate with optimizer via `scheduler.step()`
- **Key Implementation Detail:** Warmup prevents extreme gradients at initialization
- **Tests:** Learning rate progression, warmup behavior, cosine scheduling (4+ tests)

#### **4.4 Checkpoint Manager**
- **File:** `src/fundamentallm/training/checkpoint.py`
- **Classes:**
  - `CheckpointManager`: Save/restore training state
    - Attributes: model state, optimizer state, scheduler state, epoch, step, metrics
    - Methods:
      - `save(path, model, optimizer, scheduler, metrics={})`: Full checkpoint
      - `load(path) → (model, optimizer, scheduler, metrics, epoch, step)`: Restore all
      - `save_best(path, model, metrics)`: Keep best checkpoint (lowest val loss)
      - `save_last(path, model, metrics)`: Keep last N checkpoints (default 3)
    - Support for resume training (no epoch/step reset)
  - Error handling: Graceful checkpoint corruption detection
- **Key Implementation Detail:** Use `torch.save()` / `torch.load()` for portability
- **Tests:** Save/load cycle, best checkpoint selection, resume training (3+ tests)

#### **4.5 Trainer Class**
- **File:** `src/fundamentallm/training/trainer.py`
- **Class:**
  - `Trainer`: Main training loop orchestrator
    - **Constructor:** model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, config
    - **Methods:**
      - `train_epoch() → loss`: Single epoch training with gradient accumulation
      - `validate() → metrics`: Validation loop (no grad, no weight update)
      - `train(num_epochs, checkpoint_dir) → History`: Full training with checkpoint saving
      - `_train_step(batch) → loss`: Single batch forward/backward
      - `_validate_step(batch) → loss`: Single batch validation
    - **Features:**
      - Gradient accumulation (accumulation_steps): Simulates larger batches
      - Gradient clipping (max_norm=1.0): Stabilizes long sequences
      - EMA (exponential moving average) for loss smoothing
      - Progress tracking: epoch, step, batch loss, val loss
      - Device management: Automatic .to(device) for tensors
      - Mixed precision (optional, using torch.autocast for inference; training with AMP if enabled)
    - **Callbacks integration:** Hooks for on_epoch_end, on_validation, on_checkpoint
    - **Metrics tracking:**
      - Training loss (per batch, per epoch, EMA)
      - Validation loss (per epoch)
      - Perplexity = exp(loss)
      - Throughput (tokens/sec, samples/sec)
  - **Error handling:** NaN detection, checkpoint rollback on NaN
- **Key Implementation Detail:** Gradient clipping prevents explosion on long sequences; EMA smooths noisy loss curves
- **Tests:** Training loop convergence, validation, gradient accumulation, NaN handling, checkpoint integration (5+ tests)

#### **4.6 Training Configuration**
- **File:** Update `src/fundamentallm/config/__init__.py`
- **Class:** `TrainingConfig` (already exists, ensure all fields present)
  - **Fields:**
    - learning_rate: float = 1e-4
    - optimizer: str = "adamw"  # adamw, adam, sgd, rmsprop
    - optimizer_eps: float = 1e-8
    - optimizer_weight_decay: float = 0.01
    - scheduler: str = "linear_warmup"  # constant, linear_warmup, cosine, exponential
    - warmup_steps: int = 500
    - total_steps: int (computed from num_epochs × train_steps_per_epoch if None)
    - max_grad_norm: float = 1.0
    - accumulation_steps: int = 1
    - gradient_checkpointing: bool = False  # For memory efficiency
    - mixed_precision: bool = False  # fp16 training (optional for v1)
    - num_epochs: int = 3
    - eval_steps: int = 100  # Validation every N steps
    - checkpoint_dir: str = "./checkpoints"
    - device: str = "cpu"  # "cuda:0", "mps" on Mac
    - seed: int = 42
  - **Validation:** ensure total_steps is computed if None, max_grad_norm > 0
- **Key Implementation Detail:** Config drives entire trainer behavior; YAML loading enables reproducibility

#### **4.7 Integration Tests**
- **File:** `tests/integration/test_training_pipeline.py`
- **Coverage:**
  - End-to-end training: tokenize → dataset → loader → train/val → checkpoint
  - Training loop: convergence on toy dataset (small model, 1-2 epochs)
  - Gradient accumulation: same result as larger batch size
  - Checkpoint cycle: train → save → load → resume → verify loss continues decreasing
  - Validation metrics: loss decreases, perplexity decreases
  - NaN detection: corrupted input handling
  - Scheduler integration: learning rate changes over epochs
- **Tests:** 6+ integration tests
- **Toy Dataset:** Small Shakespeare corpus (1000 chars, vocab=50) for fast training

---

## Implementation Order

1. **Loss Computation** (4.1) - Simplest, no dependencies on trainer
2. **Optimizer Builder** (4.2) - Factory pattern, independent
3. **Learning Rate Scheduler** (4.3) - Depends on optimizer interface
4. **Checkpoint Manager** (4.4) - Depends on model/optimizer/scheduler
5. **Trainer Class** (4.5) - Integrates all above components
6. **Training Config** (4.6) - Update config to include all fields
7. **Integration Tests** (4.7) - Test full pipeline

---

## Key Design Patterns

### Gradient Accumulation
```python
# Simulates larger batch by accumulating gradients before step()
for i, batch in enumerate(loader):
    loss = loss_fn(model(batch['input_ids']), batch['target_ids'])
    loss = loss / accumulation_steps  # Scale for averaging
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Gradient Clipping
```python
# Prevents exploding gradients on long sequences
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Validation Loop (No Gradients)
```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        loss = loss_fn(model(batch['input_ids']), batch['target_ids'])
        val_losses.append(loss.item())
model.train()  # Reset to training mode
```

### Training/Validation Metrics
```python
metrics = {
    "loss": mean(losses),
    "perplexity": exp(mean(losses)),
    "throughput_tokens_per_sec": total_tokens / elapsed_time,
    "lr": current_lr,
}
```

---

## Testing Strategy

### Unit Tests (10+ tests)
- Loss computation: shape handling, label smoothing, gradient flow
- Optimizer builder: all optimizer types, parameter groups
- Schedulers: warmup, cosine decay, exponential decay progressions
- Checkpoint manager: save/load cycle, best/last selection
- Trainer: single batch update, validation step, gradient accumulation

### Integration Tests (6+ tests)
- Full training loop on toy dataset (1-2 epochs)
- Resume from checkpoint and verify loss continues
- Validation metrics monotonically improve
- Gradient accumulation gives same results as larger batch
- NaN handling and recovery
- Scheduler integration (LR decreases over time)

### Edge Cases
- Empty validation loader
- Single sample per batch
- Very long sequences (test gradient clipping)
- Multiple validation steps per epoch

---

## Reference Implementation Points

### From Phase 3 (Adapt These Patterns)
- **Config validation:** Use Pydantic validators for device, optimizer names, scheduler types
- **Module pattern:** All components inherit from `nn.Module` or standalone utilities
- **Type hints:** Full type hints with `from __future__ import annotations`
- **Error messages:** Informative failures (e.g., "warmup_steps must be < total_steps")

### From Phase 2 (Adapt These Patterns)
- **Callbacks:** Integrate with callback system from Phase 1 for extensibility
- **Config I/O:** TrainingConfig loads from YAML, saves with `to_yaml()`
- **Fixtures:** Use pytest fixtures for trainer + dataloaders in tests

---

## Success Criteria

- ✅ All 6 components implemented with full type hints
- ✅ 10+ unit tests passing
- ✅ 6+ integration tests passing
- ✅ Toy dataset training converges (loss decreases) after 2 epochs
- ✅ Checkpoint save/load/resume cycle works
- ✅ Full test suite (130 Phase 1-3 + new Phase 4) passes
- ✅ Code is documented with docstrings and examples
- ✅ No warnings (except benign PyTorch numpy deprecations)

---

## Quick Start Commands

```bash
# Run all Phase 4 unit tests
pytest tests/unit/test_losses.py tests/unit/test_optimizers.py tests/unit/test_schedulers.py tests/unit/test_checkpoint.py tests/unit/test_trainer.py -v

# Run integration tests
pytest tests/integration/test_training_pipeline.py -v

# Full validation (all phases)
pytest tests/ -q

# Train on toy dataset (after Phase 4 complete)
python -m fundamentallm.training.train --config configs/train.yaml --data-path data/tiny.txt
```

---

## Important Reminders

1. **Gradient shapes:** logits [batch, seq_len, vocab_size] vs targets [batch, seq_len]; reshape both to 1D for cross_entropy
2. **NaN safeguard:** Detect NaN in loss early; checkpoint rollback prevents corrupted models
3. **Validation frequency:** eval_steps controls validation frequency (every N training steps, not epochs)
4. **Device handling:** All tensors must be on same device; `.to(device)` in trainer
5. **Test on small data:** Use toy dataset for fast iteration; only test on full data once pipeline works

---

**Ready to begin Phase 4!** Follow implementation order above, ensure each component has comprehensive tests before moving to next. Good luck! 🚀
