# Data Preparation

Learn how to prepare training data for optimal language model performance.

## Overview

**Good data = good model.** Even the best architecture won't save you from bad data.

This guide covers:
- Data cleaning and preprocessing
- Format requirements
- Splitting strategies
- Best practices

## Quick Start

### Basic Format

FundamentaLLM expects **plain text files**:

```bash
# Single file
fundamentallm train my_text.txt

# Multiple files (will be concatenated)
fundamentallm train file1.txt file2.txt file3.txt
```

### Minimal Example

```text
This is my training data.
It can have multiple lines.
The model will learn to continue patterns like this.
```

That's it! The system handles tokenization automatically.

## Data Collection

### Sources

**Good sources:**
- Books (Project Gutenberg)
- Wikipedia dumps
- Web scrapes
- Code repositories
- Domain-specific documents

**Considerations:**
- License/copyright
- Data quality
- Diversity
- Size

### How Much Data?

| Goal | Data Size | Training Time |
|------|-----------|---------------|
| Quick test | 100 KB | Minutes |
| Learn basics | 1 MB | Hours |
| Decent quality | 10+ MB | Hours to days |
| Production | 100+ MB | Days |

**FundamentaLLM samples:**
```
shakespeare25k.txt:    25 KB  (tiny)
shakespeare100k.txt:  100 KB  (small)
shakespeare1mil.txt:    1 MB  (medium)
```

## Cleaning Data

### Common Issues

#### 1. Encoding Problems

```bash
# Check encoding
file -i my_data.txt
# Should show: charset=utf-8 or charset=us-ascii

# Convert to UTF-8 if needed
iconv -f ISO-8859-1 -t UTF-8 my_data.txt > clean_data.txt
```

#### 2. Weird Characters

```python
import re

def clean_text(text):
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text
```

#### 3. HTML/Markup

```python
from bs4 import BeautifulSoup

def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()
```

#### 4. Duplicate Content

```python
def remove_duplicates(lines):
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique
```

### Cleaning Script Example

```python
#!/usr/bin/env python3
"""Clean text data for training."""
import re
import sys

def clean_text(text):
    # Remove control characters except newline/tab
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove excessive newlines (keep paragraph structure)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    clean = clean_text(text)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(clean)
    
    print(f"Cleaned {len(text):,} → {len(clean):,} characters")
```

Usage:
```bash
python clean.py raw_data.txt clean_data.txt
```

## Data Format

### Plain Text

**Recommended.** Just raw text:

```text
The quick brown fox jumps over the lazy dog.
It was the best of times, it was the worst of times.
To be or not to be, that is the question.
```

### Structured Text

Can include structure if relevant to your task:

```text
Title: Introduction
This is the introduction paragraph.

Section 1: Background
Here we discuss the background.

Section 2: Methods
The methodology is described here.
```

### Code

For code generation:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Tip:** Keep consistent indentation and formatting.

### Multiple Documents

**Option 1:** Concatenate with clear separators

```text
<|document|>
First document content...

<|document|>
Second document content...
```

**Option 2:** Use multiple files (recommended)

```bash
fundamentallm train doc1.txt doc2.txt doc3.txt
```

System automatically handles boundaries.

## Train/Validation Split

### Why Split?

**The core principle:** You can't trust a model's performance on data it's seen before!

**Analogy:** If you study using only practice problems, then take an exam with the same problems, your score doesn't tell you if you actually learned the material. It just tells you that you memorized those specific problems.

**Training set:** The practice problems the model learns from
- **Purpose:** Update model weights to minimize loss
- **Used during:** Every training step
- **Size:** Typically 80-90% of data

**Validation set:** New problems to test understanding
- **Purpose:** Measure how well the model generalizes to unseen data
- **Used during:** After each epoch to check progress
- **Size:** Typically 10-20% of data

**Why validation loss matters:**
```
Good model: train_loss = 1.5, val_loss = 1.6
  → Similar losses mean good generalization ✅

Bad model:  train_loss = 0.8, val_loss = 2.5
  → Huge gap means memorizing training data ❌
```

### The Default Split (80/20)

FundamentaLLM uses **80% training / 20% validation** by default:

```bash
# Default: 80% train, 20% validation  
fundamentallm train data.txt

# Or explicitly set:
fundamentallm train data.txt --val-split 0.2
```

**Why 80/20?**
- Standard practice in machine learning
- 80% provides enough training signal
- 20% provides reliable validation metrics

### Adjusting the Split

**For small datasets** (< 1 MB):
```bash
# Use 90/10 to maximize training data
fundamentallm train data.txt --val-split 0.1
```

**For large datasets** (> 100 MB):
```bash
# Can afford more validation data for better evaluation
fundamentallm train data.txt --val-split 0.3
```

### Automatic Split

```bash
# 80% train, 20% validation (default)
fundamentallm train data.txt

# Or customize split
fundamentallm train data.txt --val-split 0.1
```

### Manual Split

```bash
# Split yourself for more control
head -n 90000 all_data.txt > train.txt
tail -n 10000 all_data.txt > val.txt

# Train
fundamentallm train train.txt --validation-data val.txt
```

### Split Strategies

#### Random Split (Default)

```python
import random

lines = data.split('\n')
random.shuffle(lines)

split_idx = int(len(lines) * 0.9)
train = lines[:split_idx]
val = lines[split_idx:]
```

**Good for:** Most cases

#### Sequential Split

```python
# Last 10% for validation
split_idx = int(len(data) * 0.9)
train = data[:split_idx]
val = data[split_idx:]
```

**Good for:** Time series, preserving document structure

#### Document-level Split

```python
# Keep entire documents together
train_docs = documents[:int(len(documents) * 0.9)]
val_docs = documents[int(len(documents) * 0.9):]
```

**Good for:** When documents have internal structure

## Data Augmentation

### Basic Techniques

#### 1. Case Variation

```python
# Mix upper/lower case
variations = [
    "Original text",
    "ORIGINAL TEXT",
    "original text",
]
```

#### 2. Whitespace Variation

```python
# Different spacing styles
"Hello world"
"Hello  world"  # Double space
"Hello\tworld"  # Tab
```

#### 3. Back-translation

```python
# Translate to another language and back
# Introduces paraphrasing
```

### For Character Models

Less augmentation needed since character-level inherently robust.

## Data Quality Checks

### Statistics

```python
def analyze_data(text):
    print(f"Total characters: {len(text):,}")
    print(f"Total lines: {text.count(chr(10)):,}")
    
    # Vocabulary
    vocab = set(text)
    print(f"Unique characters: {len(vocab)}")
    print(f"Character range: {min(vocab)} to {max(vocab)}")
    
    # Length stats
    lines = text.split('\n')
    lengths = [len(line) for line in lines]
    print(f"Avg line length: {sum(lengths)/len(lengths):.1f}")
    print(f"Max line length: {max(lengths)}")
```

### Check Script

```python
#!/usr/bin/env python3
"""Analyze training data quality."""
import sys
from collections import Counter

def analyze(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    print(f"=== Analysis: {filepath} ===")
    print(f"Size: {len(text):,} chars ({len(text)/1024/1024:.2f} MB)")
    print(f"Lines: {text.count(chr(10)):,}")
    
    # Character distribution
    char_counts = Counter(text)
    print(f"\nUnique chars: {len(char_counts)}")
    print("\nTop 10 characters:")
    for char, count in char_counts.most_common(10):
        display = repr(char)[1:-1]  # Handle special chars
        pct = count / len(text) * 100
        print(f"  {display:8s}: {count:8,} ({pct:5.2f}%)")
    
    # Check for issues
    print("\n=== Potential Issues ===")
    
    # Non-ASCII
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii:
        pct = non_ascii / len(text) * 100
        print(f"Non-ASCII chars: {non_ascii:,} ({pct:.2f}%)")
    
    # Long lines
    lines = text.split('\n')
    long_lines = [l for l in lines if len(l) > 1000]
    if long_lines:
        print(f"Lines > 1000 chars: {len(long_lines)}")
    
    # Empty lines
    empty = sum(1 for l in lines if not l.strip())
    print(f"Empty lines: {empty:,}")

if __name__ == '__main__':
    analyze(sys.argv[1])
```

Usage:
```bash
python analyze_data.py my_training_data.txt
```

## Best Practices

### 1. Start Small

```bash
# First run: tiny dataset
fundamentallm train small_sample.txt --epochs 5

# If working well: scale up
fundamentallm train full_dataset.txt --epochs 20
```

### 2. Clean Incrementally

Don't over-clean! Some "messy" data helps robustness.

**Essential cleaning:**
- ✅ Fix encoding issues
- ✅ Remove corrupted text
- ✅ Normalize line endings

**Optional cleaning:**
- Maybe: Remove duplicates
- Maybe: Fix spelling
- Probably not: Remove stopwords, lemmatize

### 3. Preserve Structure

If data has meaningful structure (paragraphs, sections), keep it:

```text
Good:
Chapter 1

This is the first chapter.
It has multiple paragraphs.

Chapter 2

This is the second chapter.

Bad:
Chapter 1 This is the first chapter. It has multiple paragraphs. Chapter 2...
```

### 4. Diverse Data

Mix different:
- Topics
- Writing styles
- Lengths
- Time periods

Avoid:
- All one source
- Repetitive content
- Highly specialized jargon (unless that's your goal)

### 5. Check Character Distribution

```bash
# Should have normal distribution
# Punctuation: 5-10%
# Spaces: 15-20%
# Letters: 65-75%
```

Unusual distributions suggest issues.

## Domain-Specific Data

### Code

```python
# Keep structure
def my_function():
    """Docstring."""
    return 42

class MyClass:
    def __init__(self):
        self.value = 0
```

**Tips:**
- Keep indentation consistent
- Include comments/docstrings
- Mix different patterns

### Dialogue

```text
Alice: Hello, how are you?
Bob: I'm doing well, thanks!
Alice: That's great to hear.
```

**Tips:**
- Consistent speaker format
- Include context/stage directions if relevant

### Technical Writing

```text
Section 1.1: Introduction

The model is defined as follows:
  y = f(x; θ)

Where θ represents the parameters.
```

**Tips:**
- Keep section structure
- Preserve special formatting
- Include equations if relevant

## Common Pitfalls

### 1. Data Leakage

**Problem:** Validation data too similar to training.

**Fix:**
- Document-level split
- Time-based split
- Check for duplicates

### 2. Imbalanced Data

**Problem:** Too much of one type.

```text
90% news articles
10% everything else
```

**Fix:** Balance your sources.

### 3. Too Clean

**Problem:** Overly preprocessed data.

**Issue:** Model won't handle real-world messiness.

**Fix:** Keep reasonable amount of variation.

### 4. Wrong Encoding

**Problem:** Text looks like gibberish.

```text
Donâ€™t vs Don't
```

**Fix:** Ensure UTF-8 throughout pipeline.

### 5. Too Small

**Problem:** Insufficient data.

**Symptoms:**
- Model memorizes training data
- High variance between runs

**Fix:** Get more data or use smaller model.

## Advanced: Streaming Data

For very large datasets that don't fit in memory:

```python
class StreamingDataset:
    def __init__(self, filepaths):
        self.filepaths = filepaths
    
    def __iter__(self):
        for filepath in self.filepaths:
            with open(filepath, 'r') as f:
                for line in f:
                    yield line
```

FundamentaLLM handles this automatically for large files.

## Checklist

Before training:
- [ ] Data is UTF-8 encoded
- [ ] No corrupted characters
- [ ] Reasonable size (>100 KB)
- [ ] Training/validation split
- [ ] Checked statistics
- [ ] Representative of target task
- [ ] Legal to use

## Further Reading

- [Training Guide](./training.md) - After preparing data
- [Tokenization](../concepts/tokenization.md) - How text becomes tokens
- [Quick Start](./quick-start.md) - Simple training example
- C4 dataset paper (Raffel et al., 2020)
- The Pile dataset documentation

## Next Steps

- [Training](./training.md) - Train your model
- [Hyperparameters](./hyperparameters.md) - Tune for better results
- [Evaluation](./evaluation.md) - Measure performance
