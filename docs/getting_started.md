# Getting Started with FundamentaLLM

## Overview

FundamentaLLM is an educational framework for training transformer language models from scratch. Built with PyTorch, it provides a complete pipeline from tokenization to text generation.

## Installation

### Using pip (recommended)

```bash
# Clone repository
git clone https://github.com/your-org/fundamentallm.git
cd fundamentallm

# Install in development mode
pip install -e .
```

### Using virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e .
```

### Verify Installation

```bash
fundamentallm --help
```

You should see the main CLI with `train`, `generate`, and `evaluate` commands.

## Your First Model

### Step 1: Prepare Data

Create or download a text file. For testing, use the included sample:

```bash
# Use sample data
cat data/samples/sample_data.txt
```

Or download a larger dataset:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

FundamentaLLM works with any UTF-8 text file.

### Step 2: Train a Model

Train with default settings:

```bash
fundamentallm train data/samples/sample_data.txt \
    --output-dir checkpoints/my_first_model \
    --epochs 5 \
    --batch-size 32
```

This will:
- Create a character-level tokenizer
- Build a small transformer model
- Train for 5 epochs
- Save checkpoints to `checkpoints/my_first_model/`

**Output:**
```
Training with config:
  Model: d_model=512, num_heads=8, num_layers=6
  Data: 1234 tokens, batch_size=32
  Training: 5 epochs, lr=3e-4

Epoch 1/5: 100%|██████████| 38/38 [00:05<00:00,  7.21it/s]
  train_loss=4.123, val_loss=3.987, perplexity=53.45
  
Saved checkpoint: checkpoints/my_first_model/checkpoint_epoch_1.pt
...
```

### Step 3: Generate Text

Generate text from your trained model:

```bash
fundamentallm generate checkpoints/my_first_model/final_model.pt \
    --prompt "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.8
```

**Output:**
```
Once upon a time, in a land far away, there lived a curious young wizard who spent his days studying ancient texts and practicing spells. One morning, he discovered...
```

### Step 4: Interactive Mode

For conversation-like interaction:

```bash
fundamentallm generate checkpoints/my_first_model/final_model.pt --interactive
```

**Example session:**
```
╭──────────────────────────────────────────────────╮
│ FundamentaLLM Interactive Mode                   │
│ Type /help for commands, /quit to exit          │
╰──────────────────────────────────────────────────╯

Settings:
  temperature=0.8, max_tokens=100, top_k=50, top_p=0.95

> Hello there!
Hello there! Welcome to the interactive demo. How can I assist you today?

> /set temperature=1.0
Updated setting: temperature=1.0

> Tell me a story
[Generated story appears here...]
```

## Configuration Files

For more control, use YAML configuration files:

### Create Config

```yaml
# my_config.yaml
model:
  d_model: 256
  num_heads: 4
  num_layers: 3
  d_ff: 1024
  dropout: 0.1

training:
  max_epochs: 10
  learning_rate: 1e-3
  batch_size: 16
  warmup_steps: 100
  scheduler: "cosine"
```

### Train with Config

```bash
fundamentallm train data.txt \
    --config my_config.yaml \
    --output-dir checkpoints/custom_model
```

### Use Preset Configs

FundamentaLLM includes preset configurations:

```bash
# Small model (fast training)
fundamentallm train data.txt --config configs/small.yaml

# Large model (better quality)
fundamentallm train data.txt --config configs/large.yaml
```

## Evaluation

Evaluate your model on held-out data:

```bash
fundamentallm evaluate checkpoints/my_first_model/final_model.pt \
    --data data/samples/test_data.txt \
    --json
```

**Output:**
```json
{
  "loss": 2.345,
  "perplexity": 10.43,
  "bits_per_char": 3.21
}
```

## Command Reference

### Train Command

```bash
fundamentallm train <data_path> [OPTIONS]

Options:
  --config PATH           Config file (YAML)
  --output-dir PATH       Output directory [default: checkpoints]
  --epochs INT           Training epochs [default: 10]
  --batch-size INT       Batch size [default: 32]
  --learning-rate FLOAT  Learning rate [default: 3e-4]
  --device CHOICE        Device (cpu/cuda/mps) [default: auto]
```

### Generate Command

```bash
fundamentallm generate <checkpoint_path> [OPTIONS]

Options:
  --prompt TEXT          Generation prompt
  --interactive          Interactive mode (REPL)
  --max-tokens INT       Max tokens to generate [default: 100]
  --temperature FLOAT    Sampling temperature [default: 0.8]
  --top-k INT           Top-k sampling [default: 50]
  --top-p FLOAT         Nucleus sampling [default: 0.95]
```

### Evaluate Command

```bash
fundamentallm evaluate <checkpoint_path> [OPTIONS]

Options:
  --data PATH            Test data path (required)
  --batch-size INT       Batch size [default: 32]
  --json                 Output as JSON
```

## Troubleshooting

### CUDA out of memory

**Symptoms:** RuntimeError during training
**Solutions:**
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--config configs/small.yaml`
- Use CPU instead: `--device cpu`

### Training is slow

**Symptoms:** <1 iter/sec on GPU
**Solutions:**
- Increase batch size (if memory allows)
- Verify GPU usage: `nvidia-smi`
- Check data loading isn't bottleneck
- Ensure CUDA is properly installed

### Generated text is repetitive

**Symptoms:** Model repeats same phrases
**Solutions:**
- Increase temperature: `--temperature 1.0`
- Use top-p sampling: `--top-p 0.95`
- Train longer (more epochs/data)
- Check training loss is decreasing

### Training loss not decreasing

**Symptoms:** Loss plateaus or increases
**Solutions:**
- Increase learning rate: `--learning-rate 1e-3`
- Check data quality (encoding, size)
- Reduce model complexity
- Verify gradients: Check for NaN values

### ImportError or ModuleNotFoundError

**Symptoms:** Cannot import fundamentallm
**Solutions:**
- Reinstall: `pip install -e .`
- Check virtual environment is activated
- Verify dependencies: `pip list | grep torch`

## Next Steps

- **[Training Guide](training_guide.md)** - Advanced training techniques, hyperparameter tuning, best practices
- **[Architecture](architecture.md)** - Deep dive into transformer architecture and implementation
- **[Example Notebooks](notebooks/)** - Interactive Jupyter tutorials
- **[Contributing](../CONTRIBUTING.md)** - How to contribute to FundamentaLLM

## Examples

### Example 1: Quick Character Model

```bash
# Train tiny model on sample data
fundamentallm train data/samples/sample_data.txt \
    --config configs/small.yaml \
    --epochs 3 \
    --output-dir quick_test

# Generate
fundamentallm generate quick_test/final_model.pt \
    --prompt "The quick" \
    --max-tokens 50
```

### Example 2: Shakespeare Generator

```bash
# Download data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train medium model
fundamentallm train input.txt \
    --output-dir models/shakespeare \
    --epochs 20 \
    --batch-size 64

# Interactive generation
fundamentallm generate models/shakespeare/final_model.pt --interactive
```

### Example 3: Custom Config Training

```bash
# Create custom config
cat > my_config.yaml << EOF
model:
  d_model: 384
  num_heads: 6
  num_layers: 4
training:
  max_epochs: 15
  learning_rate: 5e-4
  batch_size: 48
EOF

# Train with config
fundamentallm train my_data.txt --config my_config.yaml
```

## FAQ

**Q: How much data do I need?**  
A: Minimum 10K characters for testing. 100K-1M for decent results. 10M+ for production models.

**Q: How long does training take?**  
A: On GPU: 10-30 minutes for small models, 1-4 hours for medium, 4-12 hours for large.

**Q: Can I use this for production?**  
A: FundamentaLLM is educational. For production, consider Hugging Face Transformers or similar.

**Q: Does it support word-level tokenization?**  
A: Currently character-level only. Word/subword tokenization planned for future releases.

**Q: Can I fine-tune existing models?**  
A: Not directly. FundamentaLLM trains from scratch. Fine-tuning planned for Phase 8.

**Q: GPU required?**  
A: No, but recommended. CPU training works but is 10-50x slower.

## Resources

- **GitHub:** https://github.com/your-org/fundamentallm
- **Issues:** https://github.com/your-org/fundamentallm/issues
- **Documentation:** https://fundamentallm.readthedocs.io (coming soon)
- **Paper:** "Attention Is All You Need" (Vaswani et al., 2017)

---

**Happy Training! 🚀**
