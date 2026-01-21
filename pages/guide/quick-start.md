# Quick Start: Your First Model in 5 Minutes

Train a character-level language model and generate text.

## Prerequisites

- FundamentaLLM installed (see [Installation](./installation.md))
- Python virtual environment activated
- ~2 minutes of compute time

## Step 1: Use Sample Data

FundamentaLLM includes sample data:

```bash
# Check what we have
ls -la data/samples/

# Or use Shakespeare data
ls -la data/raw/shakespeare/
```

## Step 2: Train Your Model

```bash
# Train a small model on sample data (takes ~1 minute)
fundamentallm train data/samples/sample_data.txt \
    --output-dir my_first_model \
    --epochs 5
```

**What happens:**
- Tokenizes the input file (character-level)
- Creates training/validation datasets (80% train, 20% validation by default)
- Trains a small transformer (128 hidden dims, 2 attention heads)
- Saves checkpoints every epoch
- Logs metrics to console

**Output:**
```
╭─── Training Progress ───╮
│ Epoch 1/5              │
│ Loss: 4.234            │
│ ├─ Train: 4.234        │
│ └─ Val:   4.156        │
├─ Time: 0.32s           │
│ Checkpoint saved       │
╰────────────────────────╯
```

## Step 3: Generate Text

Your trained model is now saved in `models/my_first_model/`. The key files are:
- `models/my_first_model/best.pt` - Best model (based on validation loss)
- `models/my_first_model/final_model.pt` - Final model (after all epochs)
- `models/my_first_model/training.yaml` - Training configuration
- `models/my_first_model/tokenizer.json` - Character vocabulary

Let's generate text from it:

```bash
# Generate from your trained model
fundamentallm generate my_first_model/final_model.pt \
    --prompt "The " \
    --max-tokens 100 \
    --temperature 0.7
```

**Output example:**
```
The ancient prophecy spoke of a time when...
```

The generated text will be somewhat coherent but still primitive since we only trained for 5 epochs on small data.

## Step 4: Interactive Mode

Try the interactive chat-like interface:

```bash
fundamentallm generate my_first_model/final_model.pt --interactive
```

**Features:**
- Type prompts and get instant responses
- Commands: `/set temperature=0.8`, `/help`, `/quit`
- Each generation creates a new starting point

```
╭──────────────────────────────────────────────────╮
│ FundamentaLLM Interactive Mode                   │
│ Type /help for commands, /quit to exit          │
╰──────────────────────────────────────────────────╯

> Once upon a time
Once upon a time there lived...

> /set temperature=0.9
Updated: temperature=0.9

> The magic of
The magic of language is that it...

> /quit
Goodbye!
```

## Understanding What Happened

### The Model Architecture

```
Input Text
    ↓
Tokenizer (character → integer IDs)
    ↓
Embedding Layer (ID → vector representation)
    ↓
Transformer (6 layers of attention + feedforward)
    ↓
Output Layer (predict next character)
    ↓
Sampler (select character based on probabilities)
    ↓
Generated Text
```

### Training Process

1. **Tokenization:** "Hello" → [H, e, l, l, o]
2. **Embedding:** [H, e, l, l, o] → 128-dimensional vectors
3. **Forward pass:** Process through 6 transformer layers
4. **Prediction:** Output layer predicts next character
5. **Loss calculation:** Compare prediction to actual next character
6. **Backpropagation:** Compute gradients
7. **Update:** Adjust model weights based on gradients

### Why It Works

Even with minimal training:
- Character-level modeling captures patterns in text
- Transformers are powerful at sequence prediction
- The model learns character frequencies and simple patterns
- More training = better patterns learned

## Next Steps

### To Understand Better

1. **[Concepts: Transformers](../concepts/transformers.md)** - How the model works
2. **[Concepts: Tokenization](../concepts/tokenization.md)** - Why character-level?
3. **[Concepts: Language Modeling](../concepts/language-modeling.md)** - The learning objective

### To Train Better

1. **[Training Guide](./training.md)** - Hyperparameter tuning
2. **[Hyperparameters](./hyperparameters.md)** - Which settings matter most
3. **[Data Preparation](./data-prep.md)** - Using your own datasets

### To Explore Deeper

1. **[Modules: Models](../modules/models.md)** - Architecture details
2. **[Modules: Training](../modules/training.md)** - Training loop details
3. **[Tutorials: Training Deep Dive](../tutorials/training-deep-dive.md)** - Full walkthrough

## Common Customizations

### Train Longer with More Data

```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \
    --output-dir shakespeare_model \
    --epochs 20 \
    --batch-size 32
```

### Use Custom Hyperparameters

```bash
fundamentallm train data/samples/sample_data.txt \
    --output-dir custom_model \
    --model-dim 256 \
    --num-heads 4 \
    --num-layers 8 \
    --learning-rate 0.001 \
    --epochs 10
```

### Evaluate Model

```bash
fundamentallm evaluate my_first_model/final_model.pt data/samples/sample_data.txt
```

## Troubleshooting

**Q: Training is very slow?**
- A: You're on CPU. Install PyTorch with CUDA support for GPU acceleration.

**Q: `IndexError` during training?**
- A: Data file is too small. Try a larger dataset from `data/raw/shakespeare/`.

**Q: Generated text is nonsense?**
- A: Normal for small training! Train longer or on more data.

## What's Next?

- Read [Training Guide](./training.md) to optimize your models
- Explore [Concepts](../concepts/overview.md) to understand the theory
- Follow a [Tutorial](../tutorials/first-model.md) for detailed walkthroughs
