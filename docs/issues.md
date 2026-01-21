# Known issues

## Code
- DONE: Ensure at the end of all model traning runs, that it measures and outputs the model evaluation result
- DONE: If model training reaches completion - i.e. there is a working `final_model.pt` (and in the case, that the final model is not the best model, then also a `best.pt`) - then clean up the previous steps and epochs.
- DONE: Idea: split out the validation test set into to output directory. This allows to evaluate the model again manually.
- DONE: Change default train/test data set spilt to 80/20

## Pages
- Update the documentation, that touches the changes we have made in the code (see the points under "Code" in this document)
- Formatting of the bash command examples, that has multiple line, doesn't allow direct copy-paste
- Deeper explanation on how our code actually spilts data for model training and validation