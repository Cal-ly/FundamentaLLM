# Documentation Update Handoff

## Summary of Changes Made

This session implemented configuration validation improvements, expanded CLI support, and reorganized file structure. Below is a detailed inventory for documentation updates.

---

## 1. **Strict Model Configuration Validation with Auto-Fix**

### What Changed
- **File**: validation.py, commands.py
- Model head dimension floor reduced from 64 to 8 (more flexible for small models)
- Added automatic head adjustment: when `num_heads` is too high relative to `d_model`, the CLI now:
  - **Enabled by default**: `--auto-fix-config` automatically reduces `num_heads` to nearest safe divisor
  - **Logs adjustment**: Outputs `Warning: auto-fix adjusted num_heads from X to Y (d_model=Z, head_dim=...)`
  - **Only blocks critical issues**: Only divisibility and head errors remain fatal; other warnings (e.g., small d_model) are informational

### User Impact
- Commands that previously failed with head mismatch now succeed and adjust automatically
- Users see a clear warning about what was changed, e.g.: `Warning: auto-fix adjusted num_heads from 16 to 8 (d_model=256, head_dim=32)`
- Can disable with `--no-auto-fix-config` if desired

### Docs to Update
- **cli-overview.md**: Add section on auto-fix behavior and when it triggers
- **hyperparameters.md**: Explain head_dim = d_model / num_heads relationship, note that auto-fix keeps head_dim >= 8
- **troubleshooting.md**: Add entry about "num_heads (X) too high relative to d_model (Y)" → mention auto-fix

---

## 2. **Gradient Clipping CLI Support**

### What Changed
- **File**: commands.py
- Added `--gradient-clip FLOAT` flag to the train command
- Wires through to `TrainingConfig.gradient_clip_norm`

### User Impact
- Users can now pass `--gradient-clip 1.0` (or other values) to enable/configure gradient clipping directly from CLI
- Example: `fundamentallm train data.txt --gradient-clip 1.5`

### Docs to Update
- **cli-overview.md**: Add `--gradient-clip` to train command options list with description "Gradient clipping norm (prevents exploding gradients)"
- **hyperparameters.md**: Explain what gradient clipping does, when to use it (when loss diverges), typical range 0.5–2.0

---

## 3. **Default Dataset Split Changed to 80/20**

### What Changed
- **File**: training.py
- Changed `train_split` default from 0.9 to 0.8 (was 90/10 train/test, now 80/20)

### User Impact
- Default behavior now reserves 20% of data for validation instead of 10%
- Better model evaluation; more conservative training set
- Users can override with `--val-split 0.1` to revert to 90/10 if needed

### Docs to Update
- **training.md**: Update default split example from 90/10 to 80/20; explain why 80/20 is better for smaller datasets
- **data-prep.md**: Mention default is now 80/20 train/validation split

---

## 4. **Default Output Directory Now models**

### What Changed
- **File**: training.py, commands.py
- Default `checkpoint_dir` changed from `./checkpoints` to `./models/checkpoints`
- `--output-dir` now auto-prefixes relative paths with models:
  - `--output-dir large_model` → `./models/large_model/`
  - `--output-dir models/custom` → `./models/custom/` (no duplication)
  - `--output-dir /abs/path` → `/abs/path/` (absolute unchanged)
  - No flag → `./models/checkpoints/` (default)

### User Impact
- All model outputs organized under models folder by default; root stays clean
- Users' example commands now look like: `fundamentallm train data.txt` (checkpoints go to `models/checkpoints/`)
- Or: `fundamentallm train data.txt --output-dir my_model` (goes to `models/my_model/`)

### Docs to Update
- **cli-overview.md**: Update `--output-dir` description to note that relative paths are scoped to models by default
- **quick-start.md**: Update example commands to reflect new default (no need for `--output-dir` argument for basic case; outputs go to models)
- **quick-cmds.md**: Update all example commands to remove `--output-dir FundamentaLLM.` if not needed, or show it as optional

---

## 5. **Parameter Bounds and Constraints for Documentation**

Use these ranges when writing CLI examples and documentation:

| Parameter | Min | Max | Warn Threshold | Notes |
|-----------|-----|-----|-----------------|-------|
| `d_model` | 64 (recommended) | 8192 | <64 for warn | Must be divisible by `num_heads` |
| `num_heads` | 1 | d_model/8 | — | Must divide `d_model`; auto-fix steps down to nearest divisor |
| `num_layers` | 1 | 48 | — | Hard limit from schema |
| `sequence_length` | 1 | 8192 | >8192 for OOM warn | — |
| `batch_size` | 1 | 2048 | >2048 for OOM warn | — |
| `accumulation_steps` | 1 | ∞ | — | Ideally ≤ batch_size for efficiency |
| `learning_rate` | >0 | 0.1 | <1e-6 or >0.1 | Prefer 1e-4 to 1e-3 for examples |
| `dropout` | 0.0 | 1.0 | — | Typically 0.1–0.3 |
| `gradient_clip` | >0 | 10 | >10 for warn | Typical 0.5–2.0 |
| `epochs` | 1 | 10000 | >10000 for warn | Typical 10–50 for small models |
| `train_split` | 0 (exclusive) | 1 (exclusive) | — | Now defaults to 0.8 (80/20 train/val) |
| `device` | — | — | — | cpu \| cuda \| mps \| auto |

---

## 6. **Example Command Updates**

### Old Pattern (to be updated)
```bash
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt \
  --output-dir models/comp_model_deep \
  --epochs 50 \
  --batch-size 64
```

### New Pattern (recommended)
```bash
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt \
  --output-dir comp_model_deep \
  --epochs 50 \
  --batch-size 64
```
*Outputs to `models/comp_model_deep/` automatically*

### With Gradient Clip
```bash
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt \
  --output-dir comp_model_deep \
  --model-dim 512 \
  --num-heads 8 \
  --epochs 50 \
  --batch-size 64 \
  --gradient-clip 1.0
```

---

## 7. **Files to Update in pages/**

1. **cli-overview.md**
   - Add `--gradient-clip` option to train command table
   - Explain `--output-dir` now auto-prefixes with models
   - Remove or simplify examples that used explicit models prefix

2. **quick-start.md**
   - Update default checkpoint location from `./checkpoints` to `./models/checkpoints`
   - Show new example: `fundamentallm train data.txt` → checkpoints at `models/checkpoints/`

3. **hyperparameters.md**
   - Add section on `--gradient-clip` with typical values (0.5–2.0)
   - Add table showing head_dim requirements (keep ≥8, auto-fix ensures this)
   - Explain auto-fix behavior for num_heads/d_model conflicts

4. **training.md**
   - Update default train/test split from 90/10 to 80/20
   - Explain why 80/20 is better for validation
   - Note that auto-fix may adjust model params (with warning logged)

5. **troubleshooting.md**
   - Add: "num_heads (X) too high relative to d_model (Y)" → auto-fix reduces heads, logs warning
   - Add: How to disable auto-fix if needed (`--no-auto-fix-config`)

6. **quick-cmds.md**
   - Update all command examples to use new defaults
   - Show that `--output-dir` now auto-prefixes: `--output-dir my_model` → `models/my_model/`

---

## 8. **New/Enhanced Content Areas**

### Auto-Fix Explanation (new section)
Write a brief explanation of why auto-fix is enabled by default and what it does:
- Prevents training from failing on valid but slightly misconfigured head settings
- Logs clear warnings so users understand what changed
- Can be disabled with `--no-auto-fix-config` for strict validation

### Parameter Bounds Table
Create or update a parameter bounds reference table (use table in section 5 above) for quick user lookup.

### Gradient Clipping Guide
Brief subsection explaining:
- What gradient clipping prevents (exploding gradients, NaN loss)
- When to use (loss diverges, spikes to NaN)
- Typical values (0.5–2.0)

---

## 9. **Testing Coverage Notes**

All changes are tested:
- CLI integration tests pass (181 tests total)
- Auto-fix behavior verified
- Output directory logic verified
- Gradient clip wiring verified

---

## Handoff Checklist

- [ ] Update cli-overview.md with new CLI options and auto-prefix behavior
- [ ] Update quick-start.md with new defaults and examples
- [ ] Update hyperparameters.md with parameter bounds and auto-fix explanation
- [ ] Update training.md with 80/20 split and auto-fix notes
- [ ] Update troubleshooting.md with auto-fix troubleshooting
- [ ] Update quick-cmds.md to reflect new defaults
- [ ] Add or enhance parameter bounds reference table
- [ ] Add gradient clipping guide section