# Known issues

## Code
- DONE: Ensure at the end of all model traning runs, that it measures and outputs the model evaluation result
- DONE: If model training reaches completion - i.e. there is a working `final_model.pt` (and in the case, that the final model is not the best model, then also a `best.pt`) - then clean up the previous steps and epochs.
- DONE: Idea: split out the validation test set into to output directory. This allows to evaluate the model again manually.
- DONE: Change default train/test data set spilt to 80/20
- Handle issues like conflicting parameters. Here's an example:
    ```bash
    2026-01-21 12:31:16 - fundamentallm.config.validation - WARNING - TransformerConfig validation found 1 issue(s):
    2026-01-21 12:31:16 - fundamentallm.config.validation - WARNING -   - num_heads (16) too high relative to d_model (512), d_model//num_heads would be < 64
    ```


## Pages

### Content
- Update the documentation, that touches the changes we have made in the code (see the points under "Code" in this document)
- With the focus on "training", ensure to explain - and if possible use analogies - on what differet parameters does and how it affects the model. E.g. what does `--batch-size` do and how does it affect model training and the final model.
- Ensure there is an explanation for what we output during training. E.g. go deep into what `train_loss=2.933890 | val_loss=2.508849 | lr=1.37e-04 | throughput=98091 tokens/sec` is interpreted and what does it signify on a conceptual/theoretical level. 
- Deeper explanation on how our code actually spilts data for model training and validation, and why it is important and what model evalutation can give us.
- Explan in depth, why there might be a difference between a final model and a best model, in regards to a model trained on the same material and with the same parameters.

### Formatting
- DONE: Formatting of the bash command examples, that has multiple line, doesn't allow direct copy-paste. Start with `cli-overview.html`.
- DONE: Mathematical notation in e.g. `autoregressive.html` doens't render corrently. Currently it looks like this `$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$$`. The issue might be with the formatting itself or if we need a math or a tex package.

