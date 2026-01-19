# Phase 4: Training System

**Objective:** Implement complete training orchestration including optimizer builders, learning rate schedulers, early stopping, callbacks, metrics tracking, checkpointing, and gradient clipping.

**Status:** Planning

**Dependencies:** Phase 1 (Core Infrastructure) ✅, Phase 2 (Data Pipeline) ✅, Phase 3 (Model Architecture) ✅

**Estimated Timeline:** 3-4 days

---

## Overview

Phase 4 is the training engine of FundamentaLLM. This includes:
- Optimizer builders (AdamW, Adam, SGD)
- Learning rate schedulers (cosine, linear, constant, warmup)
- Early stopping mechanism with metric tracking
- Callback system for extensibility
- Metrics tracking and history
- Checkpoint manager for model saving/loading
- Gradient clipping for stability
- Mixed precision training support
- Progress bars and comprehensive logging
- Main Trainer class orchestrating everything

This phase brings everything together to actually train models.

---

## Architecture Overview

```
Training Loop
    ├── DataLoader (from Phase 2)
    ├── Model (from Phase 3)
    ├── Optimizer (created in Phase 4.2)
    ├── LR Scheduler (created in Phase 4.3)
    ├── Loss Function (CrossEntropyLoss)
    ├── Early Stopping (created in Phase 4.4)
    ├── Callbacks (created in Phase 4.5)
    ├── Metrics (created in Phase 4.6)
    ├── Checkpointer (created in Phase 4.7)
    └── Mixed Precision (optional)
```

---

## Files to Create

### Core Training

```
src/fundamentallm/training/
├── __init__.py                     # Training module exports
├── trainer.py                      # Main Trainer class
├── optimizers.py                   # Optimizer builders
├── schedulers.py                   # LR scheduler builders
├── early_stopping.py               # EarlyStopping mechanism
├── callbacks.py                    # Callback system (enhanced from Phase 1)
├── metrics.py                      # MetricTracker class
└── loss.py                         # Loss function utilities
```

### Utilities

```
src/fundamentallm/utils/
├── checkpoint.py                   # CheckpointManager class
└── (logging.py, random.py already created in Phase 1)
```

### Testing

```
tests/
├── unit/
│   ├── test_optimizers.py          # Optimizer tests
│   ├── test_schedulers.py          # Scheduler tests
│   ├── test_early_stopping.py      # Early stopping tests
│   ├── test_callbacks.py           # Callback tests
│   ├── test_metrics.py             # Metrics tests
│   └── test_checkpointing.py       # Checkpoint tests
└── integration/
    ├── test_training_loop.py       # Full training pipeline
    └── test_training_stability.py  # Numerical stability tests
```

---

## Detailed Tasks

### Task 4.1: Optimizer Builders

**Objective:** Create flexible optimizer instantiation

**File:** `src/fundamentallm/training/optimizers.py`

**Function: `create_optimizer()`**

```python
def create_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer from config."""
```

**Supported Optimizers:**

1. **AdamW** (Recommended)
   - Decoupled weight decay (better regularization)
   - Parameters: lr, weight_decay, betas, eps
   - Good for large models
   
   ```python
   if optimizer_type.lower() == "adamw":
       return torch.optim.AdamW(
           model.parameters(),
           lr=learning_rate,
           betas=(kwargs.get("adam_beta1", 0.9), 
                  kwargs.get("adam_beta2", 0.999)),
           eps=kwargs.get("adam_epsilon", 1e-8),
           weight_decay=weight_decay
       )
   ```

2. **Adam**
   - Traditional adaptive learning rate
   - Parameters: lr, betas, eps
   - Older alternative
   
   ```python
   elif optimizer_type.lower() == "adam":
       return torch.optim.Adam(
           model.parameters(),
           lr=learning_rate,
           betas=(kwargs.get("adam_beta1", 0.9),
                  kwargs.get("adam_beta2", 0.999)),
           eps=kwargs.get("adam_epsilon", 1e-8),
           weight_decay=weight_decay
       )
   ```

3. **SGD**
   - Simple gradient descent with momentum
   - Parameters: lr, momentum
   - Baseline for comparison
   
   ```python
   elif optimizer_type.lower() == "sgd":
       return torch.optim.SGD(
           model.parameters(),
           lr=learning_rate,
           momentum=kwargs.get("momentum", 0.9),
           weight_decay=weight_decay
       )
   ```

**Validation:**
- Raise error for unknown optimizer
- Validate learning_rate > 0
- Validate weight_decay >= 0

**Success Criteria:**
- ✅ Can create AdamW with correct parameters
- ✅ Can create Adam with correct parameters
- ✅ Can create SGD with correct parameters
- ✅ Raises errors for invalid inputs
- ✅ All optimizers work with backward pass

---

### Task 4.2: Learning Rate Schedulers

**Objective:** Implement flexible LR scheduling

**File:** `src/fundamentallm/training/schedulers.py`

**Function: `create_scheduler()`**

```python
def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    max_epochs: int,
    steps_per_epoch: int,
    warmup_steps: int = 0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
```

**Supported Schedulers:**

1. **Constant LR**
   - No scheduling, constant learning rate
   - Good for debugging
   
   ```python
   if scheduler_type.lower() == "constant":
       return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
   ```

2. **Linear Decay**
   - Linearly decay LR to zero
   - Formula: lr = initial_lr * (1 - step / total_steps)
   
   ```python
   elif scheduler_type.lower() == "linear":
       total_steps = max_epochs * steps_per_epoch
       return torch.optim.lr_scheduler.LinearLR(
           optimizer,
           start_factor=1.0,
           end_factor=0.0,
           total_iters=total_steps
       )
   ```

3. **Cosine Annealing** (Recommended)
   - Cosine decay from initial LR to min_lr
   - Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * step / total_steps))
   
   ```python
   elif scheduler_type.lower() == "cosine":
       total_steps = max_epochs * steps_per_epoch
       min_lr = initial_lr * kwargs.get("min_lr_ratio", 0.1)
       return torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer,
           T_max=total_steps,
           eta_min=min_lr
       )
   ```

4. **Cosine with Warm Restarts** (Advanced)
   - Restart cosine schedule periodically
   - For exploration, can help escape local minima
   
   ```python
   elif scheduler_type.lower() == "cosine_with_restarts":
       return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
           optimizer,
           T_0=kwargs.get("T_0", max_epochs // 2),
           eta_min=initial_lr * kwargs.get("min_lr_ratio", 0.1)
       )
   ```

**Warmup (Linear)**

Most modern schedulers use linear warmup at start:
```python
def add_warmup(scheduler, warmup_steps: int):
    """Wrap scheduler with linear warmup."""
    if warmup_steps > 0:
        return SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.0, end_factor=1.0, 
                         total_iters=warmup_steps),
                scheduler
            ],
            milestones=[warmup_steps]
        )
    return scheduler
```

**Typical Configuration:**
- Warmup: 100-1000 steps (0.5-5% of training)
- Cosine annealing: rest of training
- Min LR ratio: 0.01-0.1 of initial LR

**Success Criteria:**
- ✅ All schedulers can be created
- ✅ LR decreases over time
- ✅ Warmup works if configured
- ✅ Scheduler.step() works after backward

---

### Task 4.3: Early Stopping

**Objective:** Implement early stopping with patience

**File:** `src/fundamentallm/training/early_stopping.py`

**Class: EarlyStopping**

Purpose: Stop training when validation metric plateaus

```python
class EarlyStopping:
    """Early stopping based on validation metric."""
    
    def __init__(
        self,
        patience: int,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            metric: Name of metric to monitor
            mode: "min" for loss (lower better), "max" for accuracy (higher better)
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        # State
        if mode == "min":
            self.best_value = float('inf')
            self.is_better = lambda x, y: x < y - min_delta
        else:
            self.best_value = float('-inf')
            self.is_better = lambda x, y: x > y + min_delta
        
        self.counter = 0
        self.is_best = False
    
    def step(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
        
        return self.counter >= self.patience
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        if self.mode == "min":
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')
        self.is_best = False
```

**Usage Pattern:**

```python
early_stopping = EarlyStopping(patience=5, mode="min")

for epoch in range(max_epochs):
    # ... training and validation ...
    
    val_loss = validate()
    
    if early_stopping.step(val_loss):
        print("Early stopping triggered")
        break
    
    if early_stopping.is_best:
        save_best_model()
```

**Design Notes:**
- `min_delta`: Avoid stopping on noise/fluctuation
- `counter`: Epochs without improvement
- `is_best`: Flag for saving best checkpoint
- Flexible mode: works for any metric

**Success Criteria:**
- ✅ Stops when patience exceeded
- ✅ Tracks best value correctly
- ✅ is_best flag works
- ✅ Works for both min and max modes

---

### Task 4.4: Callback System (Enhanced)

**Objective:** Expand callback system for extensibility

**File:** `src/fundamentallm/training/callbacks.py` (update from Phase 1)

**Enhance Base Callback Class:**

```python
class Callback(ABC):
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called before training starts."""
        pass
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Called after training ends."""
        pass
    
    def on_epoch_begin(self, trainer: "Trainer") -> None:
        """Called at start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Called at end of each epoch."""
        pass
    
    def on_step_end(self, trainer: "Trainer", loss: float) -> None:
        """Called after each training step."""
        pass
    
    def on_validation_end(self, trainer: "Trainer", metrics: Dict) -> None:
        """Called after validation."""
        pass
```

**Built-in Callback Implementations:**

1. **LoggingCallback**
   - Log metrics at intervals
   - Track memory usage
   - Monitor LR changes

2. **SaveBestCallback**
   - Automatically save best model
   - Based on validation metric

3. **VisualizationCallback**
   - Plot training curves
   - Update in real-time (optional)

**Enhanced CallbackList:**

```python
class CallbackList:
    """Manage list of callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        """Add callback to list."""
        self.callbacks.append(callback)
    
    def remove(self, callback: Callback) -> None:
        """Remove callback from list."""
        self.callbacks.remove(callback)
    
    # ... delegate methods for all callback hooks ...
```

**Success Criteria:**
- ✅ Can add/remove callbacks
- ✅ All hooks called correctly
- ✅ Trainer reference available in callbacks
- ✅ Can customize training behavior

---

### Task 4.5: Metrics Tracking

**Objective:** Track training metrics over time

**File:** `src/fundamentallm/training/metrics.py`

**Class: MetricTracker**

```python
class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, metrics_dict: Dict[str, float]) -> None:
        """Update metrics."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get all metrics history."""
        return self.metrics.copy()
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_best(
        self,
        metric_name: str,
        mode: str = "min"
    ) -> Optional[float]:
        """Get best value for metric."""
        if metric_name not in self.metrics:
            return None
        
        if mode == "min":
            return min(self.metrics[metric_name])
        else:
            return max(self.metrics[metric_name])
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
```

**Usage:**

```python
metrics = MetricTracker()

for epoch in range(max_epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    metrics.update({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })
    
    # Query metrics
    latest_train = metrics.get_latest("train_loss")
    best_val = metrics.get_best("val_loss", mode="min")

history = metrics.get_history()
```

**Success Criteria:**
- ✅ Can track multiple metrics
- ✅ Can query history
- ✅ Can find best value
- ✅ Thread-safe (if needed)

---

### Task 4.6: Checkpoint Manager

**Objective:** Save and load model checkpoints

**File:** `src/fundamentallm/utils/checkpoint.py`

**Class: CheckpointManager**

```python
class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 3
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_history: List[Path] = []
    
    def save(
        self,
        checkpoint: Dict[str, Any],
        name: str = "checkpoint",
        is_best: bool = False
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            checkpoint: Dict with model state, optimizer state, etc.
            name: Checkpoint name (without .pt)
            is_best: If True, also save as "best.pt"
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        # Save best copy
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy(checkpoint_path, best_path)
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location="cpu")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keep last N."""
        if len(self.checkpoint_history) > self.keep_last_n:
            # Remove oldest
            oldest = self.checkpoint_history.pop(0)
            if oldest.exists() and oldest.name != "best.pt":
                oldest.unlink()
    
    def get_latest(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        return self.checkpoint_history[-1] if self.checkpoint_history else None
    
    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        return best_path if best_path.exists() else None
```

**Checkpoint Structure:**

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict() if using_amp else None,
    
    # Configuration
    "config": config,
    
    # Training state
    "epoch": current_epoch,
    "global_step": global_step,
    "best_metric": best_metric,
    
    # History
    "metrics": metrics.get_history()
}
```

**Success Criteria:**
- ✅ Can save checkpoints
- ✅ Can load checkpoints
- ✅ Old checkpoints cleaned up
- ✅ Best checkpoint tracked
- ✅ All state preserved

---

### Task 4.7: Main Trainer Class

**Objective:** Orchestrate entire training loop

**File:** `src/fundamentallm/training/trainer.py`

**Class: Trainer**

This is the core orchestrator. Key methods:

```python
class Trainer:
    """Main training orchestrator."""
    
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callback]] = None
    ):
        """Initialize trainer."""
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None
        
        # Callbacks and tracking
        self.callbacks = CallbackList(callbacks or [])
        self.metrics = MetricTracker()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode=config.early_stopping_mode
        )
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # State
        self.global_step = 0
        self.current_epoch = 0
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        self.callbacks.on_train_begin(self)
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(self)
            
            # Train
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate_epoch()
            
            # Update metrics
            self.metrics.update({
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            })
            
            # Logging
            self._log_epoch(train_metrics, val_metrics)
            
            # Callbacks
            self.callbacks.on_epoch_end(self)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")
            
            # Early stopping
            val_loss = val_metrics['loss']
            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if self.early_stopping.is_best:
                self.save_checkpoint("best", is_best=True)
        
        self.callbacks.on_train_end(self)
        return self.metrics.get_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.config.max_epochs}",
            disable=not torch.cuda.is_available()
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward
            if self.scaler:
                with autocast('cuda'):
                    logits = self.model(inputs)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
            else:
                logits = self.model(inputs)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Progress
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Step callback
            if self.global_step % self.config.log_every_n_steps == 0:
                self.callbacks.on_step_end(self, loss.item())
        
        return {
            "loss": total_loss / num_batches,
            "perplexity": torch.exp(torch.tensor(total_loss / num_batches)).item()
        }
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.scaler:
                with autocast('cuda'):
                    logits = self.model(inputs)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
            else:
                logits = self.model(inputs)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "perplexity": torch.exp(torch.tensor(total_loss / num_batches)).item()
        }
    
    def save_checkpoint(self, name: str, is_best: bool = False) -> Path:
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": self.metrics.get_history()
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        return self.checkpoint_manager.save(checkpoint, name, is_best)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint."""
        checkpoint = self.checkpoint_manager.load(path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")
```

**Key Features:**
- ✅ Complete training loop
- ✅ Validation with metrics
- ✅ Gradient clipping for stability
- ✅ Mixed precision training
- ✅ Early stopping
- ✅ Checkpoint management
- ✅ Callback integration
- ✅ Progress tracking
- ✅ Comprehensive logging

**Success Criteria:**
- ✅ Can train model for multiple epochs
- ✅ Metrics tracked correctly
- ✅ Checkpoints saved/loaded
- ✅ Early stopping works
- ✅ Backward pass works
- ✅ GPU memory managed

---

### Task 4.8: Training Tests

**Objective:** Comprehensive testing of training components

**File:** `tests/unit/test_optimizers.py`

Tests:
- ✅ All optimizers can be created
- ✅ Gradients update parameters
- ✅ Learning rate configuration works

**File:** `tests/unit/test_schedulers.py`

Tests:
- ✅ All schedulers can be created
- ✅ LR decreases over time
- ✅ Warmup works

**File:** `tests/unit/test_early_stopping.py`

Tests:
- ✅ Stops when patience exceeded
- ✅ Tracks best value correctly
- ✅ Works for min and max modes

**File:** `tests/integration/test_training_loop.py`

Tests:
- ✅ Full training loop runs
- ✅ Metrics computed correctly
- ✅ Checkpoints saved/loaded
- ✅ No NaN/Inf in loss
- ✅ Backward pass works

**Success Criteria:**
- ✅ All tests pass
- ✅ Coverage > 85%

---

### Task 4.9: Documentation and Examples

**Objective:** Document training system

**Docstrings:**
- All classes and functions documented
- Type hints complete
- Usage examples provided

**Example Training Script:** (in docstring)
```python
# Load data
train_loader, val_loader = create_dataloaders(text, tokenizer, config)

# Create model
model = Transformer(config.model)

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

# Train
metrics = trainer.train()

# Save best model
trainer.save_checkpoint("final_model", is_best=True)
```

---

## Implementation Checklist

- [ ] Create optimizer builders (Task 4.1)
- [ ] Create LR schedulers (Task 4.2)
- [ ] Create early stopping (Task 4.3)
- [ ] Enhance callback system (Task 4.4)
- [ ] Create metrics tracker (Task 4.5)
- [ ] Create checkpoint manager (Task 4.6)
- [ ] Create main Trainer class (Task 4.7)
  - [ ] _create_optimizer()
  - [ ] _create_scheduler()
  - [ ] train() main loop
  - [ ] _train_epoch()
  - [ ] _validate_epoch()
  - [ ] save_checkpoint()
  - [ ] load_checkpoint()
- [ ] Create training tests (Task 4.8)
- [ ] Add documentation (Task 4.9)

---

## Success Criteria for Phase 4

1. **Optimizer Support**
   - ✅ AdamW, Adam, SGD can be created
   - ✅ All optimizers work with backward

2. **LR Scheduling**
   - ✅ Cosine, linear, constant schedulers work
   - ✅ Warmup can be added
   - ✅ LR decreases over training

3. **Early Stopping**
   - ✅ Stops training when metric plateaus
   - ✅ Tracks best value
   - ✅ Works for any metric/mode

4. **Training Loop**
   - ✅ Can train for multiple epochs
   - ✅ Validation after each epoch
   - ✅ Checkpoints saved regularly
   - ✅ Metrics tracked

5. **Stability**
   - ✅ No NaN/Inf in training
   - ✅ Gradient clipping works
   - ✅ Mixed precision stable
   - ✅ Memory managed

6. **Testing**
   - ✅ All components tested
   - ✅ Integration tests pass
   - ✅ Coverage > 85%

---

## Next Phase Dependency

Phase 4 must be complete before starting Phase 5 (Generation & Evaluation).

All phases now depend on each other:
- Phase 1: Configuration and abstractions
- Phase 2: Data loading
- Phase 3: Model architecture
- Phase 4: Training loop ← Complete system!

---

## Performance Considerations

- **Memory**: Batch size × seq_len × d_model
- **Computation**: O(batch_size × seq_len × d_model × num_layers)
- **Mixed Precision**: ~2x memory efficiency, minimal slowdown

### Optimization Strategies
- Gradient accumulation: simulate larger batches
- Gradient checkpointing: trade memory for computation
- Mixed precision: reduce memory usage
- Distributed training: for very large models (future)

---

## Troubleshooting Guide

**Loss becomes NaN:**
- Reduce learning rate
- Increase gradient clip norm
- Reduce batch size
- Check data for anomalies

**Training too slow:**
- Increase batch size
- Use mixed precision
- Reduce num_workers if data loading is bottleneck
- Profile with PyTorch profiler

**Out of memory:**
- Reduce batch size
- Reduce sequence length
- Use mixed precision
- Reduce model size

---

## Extension Points (Future Phases)

- Distributed training (DDP)
- Different loss functions (focal loss, etc.)
- Custom metrics
- Experiment tracking (W&B, MLflow)
- Gradient accumulation scheduling
- Learning rate finder
