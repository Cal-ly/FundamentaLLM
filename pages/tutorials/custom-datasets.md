# Custom Datasets

Learn how to train FundamentaLLM models on your own data.

## Quick Start

### 1. Prepare Your Data

```bash
# Single text file
fundamentallm train my_data.txt

# Multiple files (concatenated)
fundamentallm train file1.txt file2.txt file3.txt

# That's it! No complex preprocessing needed.
```

### 2. Basic Training

```bash
fundamentallm train my_data.txt \
    --output-dir models/my_model \
    --epochs 20 \
    --batch-size 32
```

### 3. With Validation

```bash
# Split your data first
head -n 90000 my_data.txt > train.txt
tail -n 10000 my_data.txt > val.txt

# Train with validation
fundamentallm train train.txt \
    --validation-data val.txt \
    --output-dir models/my_model \
    --early-stopping
```

## Data Requirements

### Format

**Plain text files** (`.txt`), UTF-8 encoded.

```text
This is my training data.
It can have multiple lines.
Paragraphs are fine too.

The model will learn patterns from this text.
```

**That's all!** No special formatting needed.

### Size Recommendations

| Goal | Minimum Size | Recommended | Training Time |
|------|--------------|-------------|---------------|
| Quick test | 10 KB | 100 KB | Minutes |
| Learn basics | 100 KB | 1 MB | Hours |
| Quality model | 1 MB | 10 MB | Hours to day |
| Production | 10 MB | 100+ MB | Days |

**Character count guide:**
- 10 KB ≈ 10,000 characters
- 1 MB ≈ 1,000,000 characters
- 10 MB ≈ 10,000,000 characters

## Domain-Specific Examples

### Code Generation

**Prepare code dataset:**

```bash
# Collect Python files
find . -name "*.py" -type f -exec cat {} \; > code_data.txt

# Or specific project
cat src/**/*.py > project_code.txt

# Train
fundamentallm train code_data.txt \
    --output-dir models/code_model \
    --model-dim 256 \
    --num-layers 8 \
    --epochs 30
```

**Tips:**
- Keep consistent indentation
- Include comments and docstrings
- Mix different patterns
- Consider multiple languages

**Example data structure:**
```python
def factorial(n):
    """Compute factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    """Simple calculator class."""
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
```

### Dialogue/Conversational

**Format conversations:**

```text
Alice: Hello! How are you today?
Bob: I'm doing well, thank you!
Alice: That's great to hear. What have you been up to?
Bob: Just working on some projects. You?

Charlie: Hey everyone!
Alice: Hi Charlie!
Bob: Welcome!
```

**Train:**

```bash
fundamentallm train dialogue_data.txt \
    --output-dir models/dialogue_model \
    --max-seq-len 512 \
    --epochs 25
```

**Tips:**
- Consistent speaker format
- Include context
- Mix conversation styles
- Various topics

### Creative Writing

**Story format:**

```text
Chapter 1: The Beginning

Once upon a time, in a land far away, there lived a curious young explorer.
She spent her days discovering new places and meeting interesting people.

One day, she stumbled upon a mysterious cave...

Chapter 2: The Discovery

Inside the cave, ancient symbols glowed with an ethereal light.
```

**Train:**

```bash
fundamentallm train stories_data.txt \
    --output-dir models/story_model \
    --model-dim 512 \
    --num-layers 12 \
    --max-seq-len 512 \
    --temperature 0.9 \
    --epochs 30
```

**Tips:**
- Keep narrative structure
- Include diverse styles
- Mix genres
- Maintain consistency

### Technical Documentation

**Doc format:**

```text
## Installation

To install the package, run:

```
pip install mypackage
```

## Quick Start

First, import the library:

```python
import mypackage
```

Then create an instance:

```python
obj = mypackage.MyClass()
```

## API Reference

### MyClass

The main class for interacting with the library.
```

**Train:**

```bash
fundamentallm train docs_data.txt \
    --output-dir models/docs_model \
    --model-dim 256 \
    --epochs 20
```

### Scientific Text

**Academic format:**

```text
Abstract

This paper presents a novel approach to language modeling using character-level
representations. We demonstrate that transformer architectures can effectively
learn from character sequences.

1. Introduction

Language modeling has been extensively studied in natural language processing.
Traditional approaches use word-level tokenization, but character-level models
offer several advantages...

2. Methodology

Our model consists of the following components:
- Token embedding layer
- Positional encoding
- Multi-head self-attention
- Feed-forward network
```

**Train:**

```bash
fundamentallm train papers_data.txt \
    --output-dir models/academic_model \
    --model-dim 512 \
    --num-layers 10 \
    --dropout 0.15 \
    --epochs 30
```

## Data Preparation Script

### Complete Pipeline

```python
#!/usr/bin/env python3
"""Prepare custom dataset for training."""

import os
import re
import argparse
from pathlib import Path

def clean_text(text):
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize newlines (max 2 consecutive)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Remove control characters (except newline, tab)
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    
    return text.strip()

def collect_files(directory, extensions=('.txt',)):
    """Recursively collect files with given extensions."""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f'*{ext}'))
    return sorted(files)

def process_dataset(input_paths, output_path, clean=True):
    """
    Process multiple input files into single training file.
    
    Args:
        input_paths: list of input file paths
        output_path: output file path
        clean: whether to clean text
    """
    all_text = []
    
    for input_path in input_paths:
        print(f"Processing {input_path}...")
        
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        if clean:
            text = clean_text(text)
        
        all_text.append(text)
    
    # Combine with clear separators
    combined = '\n\n'.join(all_text)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Total files: {len(input_paths)}")
    print(f"Total characters: {len(combined):,}")
    print(f"Total lines: {combined.count(chr(10)):,}")
    print(f"Output: {output_path}")

def split_train_val(input_path, train_path, val_path, val_ratio=0.1):
    """
    Split data into train and validation sets.
    
    Args:
        input_path: input file
        train_path: output train file
        val_path: output validation file
        val_ratio: fraction for validation (default 0.1 = 10%)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split point
    split_idx = int(len(text) * (1 - val_ratio))
    
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Write files
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    print(f"Train: {len(train_text):,} chars → {train_path}")
    print(f"Val:   {len(val_text):,} chars → {val_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('inputs', nargs='+', help='Input files or directories')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--no-clean', action='store_true', help='Skip cleaning')
    parser.add_argument('--split', action='store_true', help='Split train/val')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_file():
            input_files.append(path)
        elif path.is_dir():
            input_files.extend(collect_files(path))
    
    if not input_files:
        print("Error: No input files found!")
        exit(1)
    
    # Process dataset
    process_dataset(
        input_files,
        args.output,
        clean=not args.no_clean
    )
    
    # Split if requested
    if args.split:
        base = Path(args.output).stem
        ext = Path(args.output).suffix
        train_path = f"{base}_train{ext}"
        val_path = f"{base}_val{ext}"
        
        split_train_val(args.output, train_path, val_path, args.val_ratio)
```

**Usage:**

```bash
# Single file
python prepare_dataset.py my_file.txt --output data/prepared.txt

# Multiple files
python prepare_dataset.py file1.txt file2.txt file3.txt --output data/prepared.txt

# Directory of files
python prepare_dataset.py data/raw/ --output data/prepared.txt

# With train/val split
python prepare_dataset.py data/raw/ --output data/prepared.txt --split

# Skip cleaning
python prepare_dataset.py my_file.txt --output data/prepared.txt --no-clean
```

## Data Augmentation

### Simple Augmentation

```python
import random

def augment_text(text, augmentation_rate=0.1):
    """
    Apply simple augmentation to text.
    
    Techniques:
    - Character deletion
    - Character duplication
    - Character swapping
    """
    chars = list(text)
    augmented = []
    
    for char in chars:
        if random.random() < augmentation_rate:
            # Augment this character
            choice = random.choice(['delete', 'duplicate', 'swap'])
            
            if choice == 'delete':
                # Skip character
                continue
            elif choice == 'duplicate':
                # Add twice
                augmented.append(char)
                augmented.append(char)
            elif choice == 'swap' and augmented:
                # Swap with previous
                augmented[-1], char = char, augmented[-1]
                augmented.append(char)
        else:
            augmented.append(char)
    
    return ''.join(augmented)

# Usage
with open('data.txt', 'r') as f:
    text = f.read()

augmented = augment_text(text, augmentation_rate=0.05)

with open('data_augmented.txt', 'w') as f:
    f.write(augmented)
```

**Note:** Character-level models are naturally robust, so minimal augmentation is usually sufficient.

## Multi-Domain Training

### Mixing Domains

```python
def create_mixed_dataset(domain_files, output_path, weights=None):
    """
    Mix multiple domain datasets with optional weighting.
    
    Args:
        domain_files: dict of {domain_name: file_path}
        output_path: output file
        weights: dict of {domain_name: weight} (optional)
    """
    if weights is None:
        weights = {domain: 1.0 for domain in domain_files}
    
    domain_texts = {}
    
    # Load all domains
    for domain, filepath in domain_files.items():
        with open(filepath, 'r') as f:
            domain_texts[domain] = f.read()
    
    # Mix with weights
    mixed = []
    for domain, text in domain_texts.items():
        weight = weights[domain]
        # Repeat based on weight
        for _ in range(int(weight)):
            mixed.append(text)
    
    # Shuffle and combine
    random.shuffle(mixed)
    combined = '\n\n'.join(mixed)
    
    # Write
    with open(output_path, 'w') as f:
        f.write(combined)
    
    print(f"Mixed dataset: {len(combined):,} chars")

# Usage
domains = {
    'code': 'data/code.txt',
    'docs': 'data/docs.txt',
    'dialogue': 'data/dialogue.txt',
}

weights = {
    'code': 2.0,      # 2x as much code
    'docs': 1.0,      # Normal amount
    'dialogue': 0.5,  # Half as much dialogue
}

create_mixed_dataset(domains, 'data/mixed.txt', weights)
```

## Common Issues

### Issue 1: Small Dataset

**Problem:** Less than 100 KB of data.

**Solutions:**
1. **Get more data** (best option)
2. **Use smaller model:**
   ```bash
   --model-dim 64 --num-layers 2
   ```
3. **More regularization:**
   ```bash
   --dropout 0.3 --weight-decay 0.1
   ```
4. **Train longer:**
   ```bash
   --epochs 50
   ```

### Issue 2: Repetitive Data

**Problem:** Dataset has lots of repetition.

**Solutions:**
1. **Remove duplicates:**
   ```python
   lines = list(set(text.split('\n')))
   text = '\n'.join(lines)
   ```
2. **Increase diversity:** Add more varied sources
3. **Use repetition penalty during generation:**
   ```bash
   --repetition-penalty 1.3
   ```

### Issue 3: Mixed Languages

**Problem:** Dataset has multiple languages.

**Solutions:**
1. **Separate by language** (usually better)
2. **Or embrace it:** Character-level handles multilingual naturally
3. **Larger model:**
   ```bash
   --model-dim 512 --num-layers 10
   ```

### Issue 4: Special Characters

**Problem:** Unicode, emojis, rare characters.

**Solution:** Character-level handles this automatically! Just ensure UTF-8 encoding.

```bash
# Check encoding
file -i my_data.txt

# Convert if needed
iconv -f LATIN1 -t UTF-8 my_data.txt > my_data_utf8.txt
```

## Workflow Examples

### Example 1: Blog Posts

```bash
# 1. Collect blog posts (Markdown files)
find ./blog -name "*.md" -exec cat {} \; > data/raw_blogs.txt

# 2. Clean (remove frontmatter, etc.)
python scripts/clean_markdown.py data/raw_blogs.txt data/clean_blogs.txt

# 3. Split
head -n 45000 data/clean_blogs.txt > data/blogs_train.txt
tail -n 5000 data/clean_blogs.txt > data/blogs_val.txt

# 4. Train
fundamentallm train data/blogs_train.txt \
    --validation-data data/blogs_val.txt \
    --output-dir models/blog_model \
    --model-dim 256 \
    --num-layers 6 \
    --epochs 25 \
    --batch-size 32

# 5. Test generation
fundamentallm generate models/blog_model/final.pt \
    --prompt "## How to Build a Blog\n\n" \
    --temperature 0.8 \
    --max-length 500
```

### Example 2: Customer Support Logs

```bash
# 1. Export logs from database
python export_logs.py --output data/support_logs.txt

# 2. Anonymize sensitive info
python anonymize.py data/support_logs.txt data/support_logs_clean.txt

# 3. Format conversations
python format_conversations.py data/support_logs_clean.txt data/formatted.txt

# 4. Train
fundamentallm train data/formatted.txt \
    --val-split 0.1 \
    --output-dir models/support_model \
    --model-dim 256 \
    --num-layers 8 \
    --epochs 30 \
    --early-stopping

# 5. Interactive mode for testing
fundamentallm interactive models/support_model/final.pt
```

## Best Practices

### 1. Data Quality > Quantity

Better to have 1 MB of clean, relevant data than 10 MB of messy data.

### 2. Validation is Essential

Always use validation set to track generalization:

```bash
--validation-data val.txt --early-stopping
```

### 3. Start Small, Scale Up

```bash
# Quick test (5 min)
--epochs 5 --model-dim 64

# If working, scale up (1 hour)
--epochs 20 --model-dim 256

# Production (1 day)
--epochs 50 --model-dim 512
```

### 4. Check Generated Output

```bash
# Generate samples frequently
fundamentallm generate model.pt --prompt "Test" --num-samples 3
```

### 5. Log Everything

```bash
--log-dir logs/experiment_1 \
--save-interval 1000 \
--eval-interval 500
```

## Further Reading

- [Data Preparation Guide](../guide/data-prep.md) - Detailed data prep
- [Training Guide](../guide/training.md) - Training process
- [Training Deep-Dive](./training-deep-dive.md) - Advanced techniques
- [Hyperparameters](../guide/hyperparameters.md) - Tuning guide

## Next Steps

- [Training Deep-Dive](./training-deep-dive.md) - Optimize training
- [Advanced Generation](./advanced-generation.md) - Better outputs
- [Evaluation](../guide/evaluation.md) - Measure quality
- [Troubleshooting](../guide/troubleshooting.md) - Fix issues
