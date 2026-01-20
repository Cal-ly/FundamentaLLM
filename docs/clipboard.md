# Train a small model on sample data (takes ~1 minute)
fundamentallm train data/raw/shakespeare/shakespeare250k.txt --output-dir models --epochs 10


# Generate from your trained model
fundamentallm generate models/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7