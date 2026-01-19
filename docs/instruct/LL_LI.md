# Lessons Learned & Lessons Identified (LL_LI)

## Phases 1-2 Summary

### Phase 1: Core Infrastructure Setup ✅
**Objective:** Project scaffolding, packaging, configuration system, base abstractions.

**Lessons Learned:**
- **File tool behavior:** The create_file tool does not accept markers like `*** End of file` in output. Rewrite entire files cleanly with terminal commands when markers appear.
- **Pydantic v2 validators:** Use `@field_validator` with `mode="before"` for path resolution in Pydantic v2+ (not deprecated `@validator`).
- **BaseModel vs Module:** Mixing ABC with torch.nn.Module works but requires careful import ordering; specify `from __future__ import annotations` to defer type hints.
- **Configuration-first design:** Separating configs (YAML) from code (Python) enables reproducibility and easy hyperparameter tuning.
- **Type hints from day one:** Full type hints (checked with mypy strict mode) caught potential issues early and improved code clarity.

**Lessons Identified:**
- Test file cleanup required extra passes when markers lingered in file tools; using `cat <<EOF` redirection is more reliable.
- License mismatch: AGPL-3.0 vs MIT discussed; stick with current AGPL unless business requires MIT.
- Module __init__ exports should be curated to avoid circular imports.

---

### Phase 2: Data Pipeline Implementation ✅
**Objective:** Tokenizer, dataset, dataloader, preprocessing.

**Lessons Learned:**
- **Tokenizer training must be tracked:** The `_trained` guard prevents silent errors when encode/decode are called before training.
- **Special token handling:** Including BOS/EOS/PAD/UNK tokens in vocab simplifies later training code; character-level tokenizers benefit from fixed special token positions.
- **Token-level splits prevent leakage:** Splitting at token level (not character level) ensures validation data never contains characters from training—critical for language modeling.
- **Batch operations:** Default batch_encode/decode implementations (in BaseTokenizer) reduce code duplication across concrete tokenizers.
- **JSON serialization:** Human-readable JSON (with `ensure_ascii=True` for consistency) is sufficient for educational tokenizers; avoid pickle for portability.
- **PyTorch Dataset striding:** Overlapping windows (stride=1) increase training samples but introduce redundancy; configurable stride lets users balance data size vs computation.
- **DataLoader best practices:** `pin_memory=True` only with GPU; `drop_last=True` on training, `False` on validation ensures stable gradient statistics.
- **Test fixture dependency order:** Fixtures that depend on other fixtures must declare them in function signature; pytest resolves order automatically.

**Lessons Identified:**
- Vocabulary size computation must account for all training corpus characters, not just a subset—adjust test assertions dynamically.
- Edge cases in datasets (short sequences, empty texts) require explicit handling with informative errors.
- PyTorch pin_memory deprecation warnings are benign but may surface in newer versions—monitor for future changes.
- Preprocessing (clean_text) is minimal; future phases might need language-specific cleaning or unicode normalization.

---

### Phase 3: Model Architecture Implementation ✅
**Objective:** Core transformer components (normalization, attention, embeddings, FFN, blocks, full model).

**Lessons Learned:**
- **RMSNorm vs LayerNorm trade-off:** RMSNorm is 50% more parameter-efficient (scale only vs weight+bias) and trains faster; LayerNorm better for reference implementations. RMSNorm is better for production models.
- **Causal masking critical:** Lower triangular mask `torch.tril(torch.ones())` prevents information leakage to future positions. Off-by-one errors cause silent failures; test with small tensors first.
- **Pre-normalization is modern standard:** Applying LayerNorm/RMSNorm *before* attention and FFN (not after) improves gradient flow and training stability significantly.
- **Weight tying reduces parameters:** Sharing `token_embedding.weight` with output projection saves ~33% parameters (critical for educational models) with no accuracy loss.
- **GPT-2 initialization crucial:** Using `N(0, 0.02)` for weight init (not PyTorch default) stabilizes training; default init is too large for deep models.
- **Causal mask shape gotchas:** Mask shape must be broadcastable to [batch, num_heads, seq_len, seq_len]; transposing or reshaping without care causes silent failures.
- **Positional encoding choice matters:** Learned encodings are more flexible but can't extrapolate; sinusoidal fixed patterns work beyond training sequence length. Learned is better for fixed-length tasks, sinusoidal for generation.
- **Attention dropout is critical:** Without dropout in multi-head attention, heads don't specialize (empirically observed); dropout value of 0.1 is standard starting point.
- **Gradient flow testing:** All components must support `.backward()`; test with small batches [2, 4, 64] to verify gradients flow through all layers.
- **Component testing in isolation:** Unit test each component (norm, attention, FFN) separately *before* integration; integration bugs are hard to debug.
- **Config-driven instantiation:** All components accept `TransformerConfig`; any hyperparameter change auto-propagates through the model without code edits.

**Lessons Identified:**
- Dropout behavior in train vs eval modes requires explicit `.eval()` in determinism tests; this is correct PyTorch behavior.
- Attention shape validation could be stricter; consider adding explicit checks for d_model % num_heads == 0.
- Sinusoidal encoding stores full positional embeddings as buffer (not learnable); for very long sequences (>10k), consider on-demand computation.
- Transformer model abstract methods (save/load) should use `state_dict()` + `load_state_dict()`; avoid pickling for portability.
- Pre-norm blocks benefit from output normalization after all blocks; standardizes activation magnitudes before output projection.

---

### Phase 4: Training System ✅
**Objective:** Losses, optimizer factory, LR schedulers, checkpointing, trainer orchestration, integration tests.

**Lessons Learned:**
- **Masked LM loss:** Explicitly zeroing ignored targets ([-100, -1]) before reduction keeps gradients clean; per-sample reduction useful for debugging.
- **Weight-decay grouping:** Biases and norm weights must be excluded from decay; parameter identity checks avoid tensor truthiness bugs.
- **Warmup + clipping:** Linear warmup paired with norm-based gradient clipping tames early-step spikes; flush pending grads when batches % accumulation ≠ 0.
- **Checkpoint strategy:** Separate best vs rolling checkpoints prevents overwriting improvements while bounding disk use.
- **Trainer ergonomics:** Accumulation, EMA smoothing, and optional AMP need a shared step helper; allowing `eval_steps=0` cleanly disables mid-epoch eval.
- **Deprecation hygiene:** Migrating to `torch.amp.autocast/GradScaler` removed AMP warnings; ensuring numpy is installed keeps torch utility warnings quiet.

**Lessons Identified:**
- Early stopping and richer metrics (throughput, perplexity per step) remain to be hooked into callbacks.
- Long-run stability still untested on GPU; add stress tests and larger toy corpora.
 - Logging/progress callbacks could surface LR, EMA loss, throughput each epoch for quicker diagnostics.

---

### Phase 5: Generation & Evaluation ✅
**Objective:** Text generation with sampling strategies, constraints, and model evaluation metrics.

**Lessons Learned:**
- **Sampler abstraction cleanly separates algorithms:** Base `Sampler` class with `sample(logits)` method enables drop-in strategy swaps; greedy/temperature/top-k/top-p share infrastructure.
- **Shape normalization prevents edge cases:** Helper functions `_prepare_logits`/`_restore_shape` handle both [vocab] and [batch, vocab] inputs transparently; avoids dimension bugs in generation loops.
- **Top-p cumulative masking tricky:** Always preserve most probable token (avoid empty distributions); normalize after filtering to prevent NaN in multinomial sampling.
- **Stop sequences require tokenizer awareness:** Encoding stop strings once during constraint init is more efficient than repeated encoding per generation step.
- **Checkpoint loading needs flexible artifact discovery:** Models can store config inline or alongside; tokenizer typically saved adjacent; fallback chain improves usability.
- **Sampler selection priority:** Explicit `sampler` kwarg > `top_k`/`top_p` > temperature; default to greedy; makes API intuitive without parameter conflicts.
- **Generation max_seq_len enforcement:** Check model's `config.sequence_length` to prevent OOM or positional encoding errors; graceful early stopping better than runtime crash.
- **Evaluation reduction matters:** Using `reduction="sum"` with manual normalization gives accurate loss; accounting for ignored tokens (-100) critical for correct perplexity.
- **Return predictions optional:** Large evaluation datasets blow up memory if predictions are always returned; make it opt-in with `return_predictions=False` default.

**Lessons Identified:**
- Temperature near zero causes div-by-zero in scaled logits; consider adding epsilon floor (temperature = max(temp, 1e-8)).
- Top-k/top-p sorting scales O(V log V) for vocab size V; acceptable for character-level (V~100) but watch for BPE (V~50k).
- Batch generation is sequential (not parallel); true batch generation requires padding and attention mask handling—defer to future.
- ModelEvaluator and TextGenerator share checkpoint loading logic; consider extracting to shared utility function.
- Beam search, nucleus sampling variants, and length penalties not implemented; extensible via custom Sampler subclasses.
- Stop sequences check full token history each step; for long generations (>1k tokens), consider suffix-only checking or trie-based matching.

---

### Phase 6: CLI & Interactive Interface ✅
**Objective:** Click-based CLI for train/generate/evaluate commands plus Rich-powered interactive REPL.

**Lessons Learned:**
- **Click is declarative and composable:** Command groups, decorators for options/arguments, and `CliRunner` make CLI testing straightforward; `click.Path(path_type=Path)` avoids manual path coercion.
- **Config merge hierarchy matters:** When loading YAML, distinguish between {model:..., training:...} and flat formats; CLI overrides win over config file, which wins over defaults.
- **Checkpoint embedding for self-contained models:** Embedding `model_config` and `training_config` inside checkpoint dict enables `from_checkpoint()` to work standalone; fallback to adjacent YAML files gives flexibility.
- **Path serialization in YAML:** pydantic `model_dump()` returns Path objects; YAML dumper can't serialize them; convert to strings in `BaseConfig.save()` for portability.
- **Rich Console for pretty output:** `Panel`, `Prompt.ask`, and formatting markup make REPL feel polished; capturing console in tests with `record=True` enables verification.
- **Interactive REPL state management:** Store generation params (temperature, top_k, etc.) as instance attributes; `/set param=value` command pattern is intuitive; history list enables replay/debug.
- **Evaluation sequence_length defaults:** When evaluating arbitrary data via CLI, hard-coded defaults (256) can cause empty datasets; use smaller defaults (32) or require user-supplied config.
- **Entry point vs module invocation:** `pyproject.toml` script entry points directly to `cli()` function; `__main__.py` wraps it in `main()` for `python -m` usage; both work seamlessly.
- **Testing CLI with CliRunner:** CliRunner.invoke() provides full CLI testing without subprocess overhead; capturing `result.output` and `result.exit_code` validates end-to-end behavior.
- **CLI arg validation via Click types:** `click.Choice` enforces device options; `click.Path(exists=True)` prevents file-not-found at parse time; shows errors before expensive setup.

**Lessons Identified:**
- Config loading has three paths (inline dict in checkpoint, adjacent YAML, explicit arg); could simplify by standardizing on single method.
- Interactive mode doesn't support multi-line prompts or continuation; could add bracket/quote detection for complex prompts.
- CLI progress bars (tqdm integration) not implemented; could show progress during training/generation for better UX.
- Evaluation command creates new DataLoader every time; could cache for repeated evals on same data.
- No `--version` test; should verify version string displays correctly.
- REPL history could be saved to disk for session persistence across runs.
- Generate command doesn't support batch prompts from file; could add `--prompt-file` for bulk generation.

---

## Key Design Decisions So Far

1. **Character-level tokenization** (not BPE): Simpler for learning, direct encoding/decoding, no merge tables.
2. **Pydantic configs with YAML I/O**: Type-safe, validated, human-readable, environment variable overrides possible.
3. **Callback-based architecture**: Flexible training hooks without modifying trainer core.
4. **Configuration paths resolved via `expanduser().resolve()`**: Cross-platform support, relative path support.
5. **Pytest fixtures over monolithic test setup:** Clear dependencies, easy to reuse, better error messages.

---

## Upcoming Phase Considerations
- Phase 5: +12 passing tests (sampling, generation, evaluation, integration pipeline).

### Phase 5: Generation
- Temperature sampling prone to numerical instability at low temps; add epsilon floor.
- Top-k/top-p require sorting large tensors; consider efficiency for large vocab.
- Beam search complexity grows exponentially; not planned for v1 but extensible via callbacks.

### Phase 6: CLI
- Click groups for subcommands (train, generate, evaluate) keep CLI modular.
- REPL mode requires history management and graceful interrupt handling (Ctrl+C).

### Phase 7: Documentation
- Example notebooks should include training on small Shakespeare subset (reproducible, fast).
- API docs auto-generated from docstrings; ensure all functions have complete docstrings.
- Architecture diagrams (ASCII art or external) help with explanation.

---

## Testing Strategy

### Current Coverage
- Phase 1: Import tests, config validation tests (in Phase 2).
- Phase 2: 18 passing tests (tokenizer, dataset, dataloader, integration).
- Phase 3: 112 passing tests (model components + transformer).
- Phase 4: +22 passing tests (losses, optimizers, schedulers, checkpoint, trainer, pipeline).
- Phase 5: +12 passing tests (sampling, generation, evaluation, integration pipeline).
- Phase 6: +7 passing tests (CLI help, interactive REPL, full train→generate→evaluate pipeline).

### Going Forward
- Aim for >85% coverage on core modules (tokenizer, model, trainer).
- Unit tests for small tensor shapes (avoid large model tests in unit suite).
- Integration tests verify full pipeline (tokenize → dataset → loader → batch).
- Edge case tests: empty inputs, very long sequences, vocab boundary conditions.

---

## Tooling Lessons
5
- **File creation pitfalls:** Avoid `*** End of file` markers; use terminal redirection if markers appear.
- **Patch tool context:** Include 3-5 lines around target to disambiguate; complex edits better done via terminal.
- **Multi-file edits:** Use multi_replace_string_in_file for independent changes; reduces tool calls and user latency.
- **Git workflow:** Periodic commits after phase completion; helps rollback if needed.

---

## Performance & Scalability Notes

- Character tokenizer is O(n) encode/decode; fine for educational use but slow for billion-char datasets.
- LanguageModelDataset stores entire token tensor in memory; should add memory-mapped variant for future.
- DataLoader num_workers: start with 4, profile to find optimal; I/O bound vs compute bound trade-off.

---

## Future Extensions (Post-v1.0)

1. **BPE Tokenizer:** Implement tokenizers.bpe.BPETokenizer with merge tables.
2. **Multi-GPU training:** DistributedDataParallel; requires checkpointing refactor.
3. **Inference optimizations:** KV cache for generation; torch.compile() on model forward.
4. **Experiment tracking:** wandb integration in callbacks.
5. **Model checkpointing:** Load model + optimizer state; resume training mid-epoch.

---
Last Updated: January 19, 2026 (after Phase 6 completion)
