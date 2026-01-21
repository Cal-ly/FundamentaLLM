# Quick Commands

## Train a small model on sample data (takes ~1 minute)
fundamentallm train data/raw/shakespeare/shakespeare25k.txt --output-dir models/small_model --epochs 10 --device cuda

## Generate from small trained model
fundamentallm generate models/small_model/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

---

## Train a large model

### Quick

#### Train
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt --output-dir models/large_model_quick --epochs 10 --device cuda

#### Generate
fundamentallm generate models/large_model_quick/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

### Deep

#### Train a large model, deeply
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt --output-dir models/large_model_deep --epochs 20 --model-dim 512 --num-heads 8 --num-layers 12 --batch-size 64 --device cuda

#### Generate form a large, deeply trained model
fundamentallm generate models/large_model_deep/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

---

## Train a complex model

### Quick

#### Train a complex model, quickly
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt --output-dir models/comp_model_quick --epochs 10 --device cuda

#### Generate from complex, quickly trained model
fundamentallm generate models/comp_model_quick/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7

### Deep

#### Train a complex model, deeply  
fundamentallm train data/raw/shakespeare/shakespeare_complete.txt --output-dir models/comp_model_deep --model-dim 512 --batch-size 64 --num-heads 16 --epochs 50 --learning-rate 0.0003 --dropout 0.2 --mixed-precision --gradient-clip 1.0 --device cuda

#### Generate from complex, deeply trained model
fundamentallm generate models/comp_model_deep/final_model.pt --prompt "The " --max-tokens 100 --temperature 0.7