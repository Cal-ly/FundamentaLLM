# Tokenization

Tokenization is the process of converting text into numbers that neural networks can process. It's the first and most fundamental step in any language model.

## The Core Problem

Neural networks work with numbers, not text. We need to convert:

```
"Hello world" → [numerical representation]
```

But how?

## Tokenization Strategies

### 1. Character-Level Tokenization

**What:** Each character is a token.

```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [72, 101, 108, 108, 111]
```

**Pros:**
- ✅ Small vocabulary (256 for ASCII, ~150k for Unicode)
- ✅ No out-of-vocabulary words
- ✅ Handles any text, any language, emojis, special characters
- ✅ Simple to implement and understand

**Cons:**
- ❌ Longer sequences (more computation)
- ❌ Model must learn to compose characters into words
- ❌ Slower to generate

**FundamentaLLM uses character-level tokenization** for educational clarity and generality.

### 2. Word-Level Tokenization

**What:** Each word is a token.

```
"Hello world" → ['Hello', 'world'] → [5234, 1089]
```

**Pros:**
- ✅ Shorter sequences
- ✅ Natural linguistic units

**Cons:**
- ❌ Huge vocabulary (50k-100k+ words)
- ❌ Out-of-vocabulary problem ("uncommonword" → ???)
- ❌ Doesn't handle misspellings or variations
- ❌ Language-specific (different rules per language)

### 3. Subword Tokenization (BPE, WordPiece)

**What:** Split into subword units based on frequency.

```
"unhappiness" → ['un', 'happiness'] → [234, 5621]
"happiness" → ['happiness'] → [5621]
```

**Pros:**
- ✅ Balanced vocabulary size (~30k tokens)
- ✅ Handles rare words by composition
- ✅ Good for multiple languages

**Cons:**
- ❌ More complex to implement
- ❌ Requires training tokenizer on data
- ❌ Still has edge cases

**Used by:** GPT (BPE), BERT (WordPiece), T5 (SentencePiece)

## Character-Level in Detail

FundamentaLLM uses **character-level tokenization** because:

1. **Simplicity** - No complex training needed
2. **Universality** - Works for any language
3. **Educational** - Easy to understand and debug
4. **Generality** - Never encounters unknown tokens

### How It Works

```python
class CharacterTokenizer:
    def __init__(self):
        # Use all printable ASCII + common Unicode
        self.chars = list(range(256))  # Extended ASCII
        self.char_to_id = {chr(i): i for i in self.chars}
        self.id_to_char = {i: chr(i) for i in self.chars}
        
    def encode(self, text: str) -> List[int]:
        """Convert text to list of integers."""
        return [self.char_to_id[c] for c in text]
    
    def decode(self, ids: List[int]) -> str:
        """Convert list of integers back to text."""
        return ''.join([self.id_to_char[i] for i in ids])
```

### Example

```python
tokenizer = CharacterTokenizer()

# Encode
text = "Hello, world!"
ids = tokenizer.encode(text)
# → [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]

# Decode
reconstructed = tokenizer.decode(ids)
# → "Hello, world!"
```

## Vocabulary Size Comparison

| Method | Typical Vocab Size | Example Model |
|--------|-------------------|---------------|
| Character | 256-300 | FundamentaLLM |
| Subword (BPE) | 30,000-50,000 | GPT-3 |
| Word | 50,000-200,000 | Traditional NLP |

Smaller vocabulary = simpler model, but longer sequences.

## Why Character-Level for Learning?

### 1. **Transparency**

You can see exactly what the model is doing:

```
Input:  "cat"
Tokens: [99, 97, 116]
ASCII:  ['c', 'a', 't']
```

No mysterious subword splits to debug.

### 2. **Generality**

Works for any text:
```
"Hello 你好 😀" → [72, 101, 108, 108, 111, 32, 228, 189, 160, 229, 165, 189, 32, 240, 159, 152, 128]
```

Emojis, Chinese, English - all just bytes.

### 3. **No Training Required**

Subword tokenizers require training on a corpus:
```bash
# BPE requires this:
train_tokenizer(corpus, vocab_size=30000)
```

Character tokenizers: ready to go immediately.

### 4. **Educational Value**

Learn how models build up understanding from smallest units:
- Characters → Character patterns → Morphemes → Words → Syntax

## The Trade-off: Sequence Length

Character tokenization makes sequences longer:

```
Word-level:      ["The", "cat", "sat"]              (3 tokens)
Character-level: ["T","h","e"," ","c","a","t"," ","s","a","t"]  (11 tokens)
```

**Impact:**
- More memory (longer sequences)
- More computation (quadratic attention)
- Slower training and generation

**Why it's okay for FundamentaLLM:**
- Educational focus (transparency > performance)
- Small to medium texts (not training on billions of tokens)
- Modern hardware can handle it

## When to Use Each

### Use Character-Level When:
- Learning/teaching language models
- Working with many languages simultaneously
- Handling lots of special characters or technical text
- Vocabulary management is too complex

### Use Subword-Level When:
- Production systems at scale
- Performance is critical
- Have resources to train tokenizer
- Standard text in 1-3 languages

### Use Word-Level When:
- Fixed domain with controlled vocabulary
- Very simple applications
- Legacy systems

## Special Tokens

Most tokenizers include special tokens:

```python
vocab = {
    '<PAD>': 0,    # Padding (for batching)
    '<UNK>': 1,    # Unknown word
    '<BOS>': 2,    # Beginning of sequence
    '<EOS>': 3,    # End of sequence
    'a': 4,
    'b': 5,
    # ... rest of vocabulary
}
```

**FundamentaLLM approach:**
- Character-level → no `<UNK>` needed (every character is known)
- Can still use `<BOS>`, `<EOS>`, `<PAD>` if needed

## Encoding Examples

### Simple Text
```python
"Hello" 
→ [72, 101, 108, 108, 111]
→ Embedded into vectors
→ Fed to transformer
```

### With Punctuation
```python
"Hello, world!"
→ [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
#   H   e    l    l    o   ,   SPACE w   o   r   l   d   !
```

### Numbers
```python
"Price: $42"
→ [80, 114, 105, 99, 101, 58, 32, 36, 52, 50]
#   P   r    i    c    e   :  SPACE $  4   2
```

Model learns that "52" represents the number forty-two by seeing character patterns.

## Tokenization in Training Pipeline

```
1. Raw text: "The cat sat on the mat"
             ↓
2. Tokenize: [84, 104, 101, 32, 99, 97, ...]
             ↓
3. Create training pairs:
   Input:  [84, 104, 101]       "The"
   Target: [104, 101, 32]       "he "
             ↓
4. Embed:   [[0.1, 0.5, ...], [0.3, 0.2, ...], ...]
             ↓
5. Model forward pass
             ↓
6. Predict next token
```

## Implementation Details

### Handling Unicode

```python
def encode_unicode(text: str) -> List[int]:
    """Encode text as UTF-8 bytes."""
    return list(text.encode('utf-8'))

def decode_unicode(ids: List[int]) -> str:
    """Decode UTF-8 bytes back to text."""
    return bytes(ids).decode('utf-8', errors='ignore')
```

### Batch Processing

```python
def batch_encode(texts: List[str], max_len: int) -> torch.Tensor:
    """Encode and pad batch of texts."""
    encoded = [encode(text) for text in texts]
    
    # Pad to same length
    padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
    
    return torch.tensor(padded)
```

## Tokenization Metrics

### Vocabulary Coverage
Percentage of text covered by vocabulary (100% for character-level).

### Average Token Length
Character-level: 1 character per token
Subword: ~4 characters per token  
Word: ~6 characters per token

### Sequence Length Multiplier
Character-level: 1x (baseline)
Subword: ~0.25x (4x shorter)
Word: ~0.15x (7x shorter)

## Key Insights

1. **Tokenization is fundamental** - Affects model architecture, training, and capabilities
2. **Trade-offs everywhere** - Vocab size vs sequence length vs complexity
3. **Character-level = universal** - Works for everything, at cost of efficiency
4. **Production uses subword** - But character-level is great for learning

## Common Mistakes

### 1. Forgetting Special Characters
```python
# Wrong: Only letters
vocab = 'abcdefghijklmnopqrstuvwxyz'

# Right: Include spaces, punctuation
vocab = 'abcdefg... SPACE.,!?'
```

### 2. Not Handling Unicode
```python
# Wrong: ASCII only
text.encode('ascii')  # Fails on "café"

# Right: UTF-8
text.encode('utf-8')  # Works for "café"
```

### 3. Inconsistent Encode/Decode
```python
# Encode and decode must be inverses
assert decode(encode(text)) == text
```

## Further Reading

- "Neural Machine Translation of Rare Words with Subword Units" (BPE paper)
- [Hugging Face Tokenizers Guide](https://huggingface.co/docs/tokenizers/)
- [SentencePiece](https://github.com/google/sentencepiece) (subword tokenizer)

## Next Steps

- [Embeddings](./embeddings.md) - Converting token IDs to vectors
- [Language Modeling](./language-modeling.md) - What we train the model to do
- [Data Module](../modules/data.md) - Implementation details
