# Known issues

## Code
- DONE: Ensure at the end of all model traning runs, that it measures and outputs the model evaluation result
- DONE: If model training reaches completion - i.e. there is a working `final_model.pt` (and in the case, that the final model is not the best model, then also a `best.pt`) - then clean up the previous steps and epochs.
- DONE: Idea: split out the validation test set into to output directory. This allows to evaluate the model again manually.
- DONE: Change default train/test data set spilt to 80/20
- DONE: Suggest how we can handle issues like conflicting parameters. Here's an example:
```bash
2026-01-21 12:31:16 - fundamentallm.config.validation - WARNING - TransformerConfig validation found 1 issue(s):
2026-01-21 12:31:16 - fundamentallm.config.validation - WARNING -   - num_heads (16) too high relative to d_model (512), d_model//num_heads would be < 64
```

### Notes for docs updates
- Auto-fix for model head settings is now enabled by default: `--auto-fix-config` will step `num_heads` down to a safe divisor so head_dim stays >=8, logging a warning with the before/after values. Only critical head/divisibility issues remain blocking; other warnings are informational.
- Parameter limits to keep CLI examples within acceptable ranges (warn thresholds in parentheses):
	- `d_model`: >=64 recommended (warn if <64). Ensure `num_heads` divides `d_model`.
	- `num_heads`: 1..(d_model/8) and must divide `d_model`; auto-fix will reduce to the nearest safe value when needed.
	- `num_layers`: 1..48 (hard limit from schema).
	- `sequence_length`: >0 (warn if >8192 for OOM risk).
	- `batch_size`: >=1 (warn if >2048). `accumulation_steps` >=1 and ideally <= `batch_size`.
	- `learning_rate`: >0 (warn if >0.1 or <1e-6); docs should prefer 1e-4 to 1e-3 in examples.
	- `dropout`: 0.0..1.0 inclusive.
	- `gradient_clip`: >0 (warn if >10).
	- `epochs`: >=1 (warn if >10000).
	- `train_split`: in (0,1); `val_split` maps to `1 - train_split` when provided.
	- `device`: cpu | cuda | mps | auto (auto picks the best available).


## Pages

### Content
- DONE: Update the documentation, that touches the changes we have made in the code (see the points under "Code" in this document)
- DONE: With the focus on "training", ensure to explain - and if possible use analogies - on what differet parameters does and how it affects the model. E.g. what does `--batch-size` do and how does it affect model training and the final model.
- DONE: Ensure there is an explanation for what we output during training. E.g. go deep into what `train_loss=2.933890 | val_loss=2.508849 | lr=1.37e-04 | throughput=98091 tokens/sec` is interpreted and what does it signify on a conceptual/theoretical level. 
- DONE: Deeper explanation on how our code actually spilts data for model training and validation, and why it is important and what model evalutation can give us.
- DONE: Explan in depth, why there might be a difference between a final model and a best model, in regards to a model trained on the same material and with the same parameters.

### Formatting
- DONE: Formatting of the bash command examples, that has multiple line, doesn't allow direct copy-paste. Start with `cli-overview.html`.
- DONE: Mathematical notation in e.g. `autoregressive.html` doens't render corrently. Currently it looks like this `$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$$`. The issue might be with the formatting itself or if we need a math or a tex package.

