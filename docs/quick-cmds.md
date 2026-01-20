# Quick Commands

## Train a small model on sample data (takes ~1 minute)
fundamentallm train data/raw/shakespeare/shakespeare25k.txt --output-dir models/small_model --epochs 10 --device cuda

## Generate from small trained model
fundamentallm generate models/small_model/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

---

## Train a large model quickly on sample data
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt --output-dir models/large_model1 --epochs 10 --device cuda

## Train a large model deeply on sample data
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt --output-dir models/large_model2 --batch-size 64 --epochs 50 --learning-rate 0.0003 --device cuda

## Generate from large, quickly trained model
fundamentallm generate models/large_model/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

## Generate from large, deeply trained model
fundamentallm generate models/large_model/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

---
## Train a complex model quickly on sample data
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt --output-dir models/comp_model1 --epochs 10 --device cuda

## Train a complex model deeply on sample data 
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt --output-dir models/comp_model2 --batch-size 64 --epochs 50 --learning-rate 0.0003 --device cuda

## Generate from complex, quickly trained model
fundamentallm generate models/comp_model1/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

## Generate from complex, deeply trained model
fundamentallm generate models/comp_model2/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7