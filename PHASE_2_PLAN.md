# Phase 2: Data Pipeline Implementation

**Objective:** Implement the complete data pipeline including tokenization, dataset creation, and data loading with proper train/validation splitting.

**Status:** Planning

**Dependencies:** Phase 1 (Core Infrastructure) ✅

**Estimated Timeline:** 2-3 days

---

## Overview

Phase 2 builds the data processing layer that feeds training data to the model. This includes:
- Character-level tokenizer with special token support
- Abstract tokenizer base class validation
- Custom PyTorch Dataset for language modeling
- DataLoader builders with proper train/validation split
- Data preprocessing utilities
- Tokenizer serialization (save/load)
- Test data and fixtures

This phase is critical because data quality and proper splitting directly impact model training success.

---

## Architecture

```
Text File (raw data)
        ↓
    Tokenizer
   (train/encode/decode)
        ↓
   Token List
        ↓
 LanguageModelDataset
        ↓
   DataLoader
        ↓
    Training Loop
```

---

## Files to Create

### Core Implementation

```
src/fundamentallm/data/
├── __init__.py                     # Data module exports
├── tokenizers/
│   ├── __init__.py
│   ├── base.py                     # BaseTokenizer (from Phase 1, refine)
│   └── character.py                # CharacterTokenizer implementation
├── dataset.py                      # LanguageModelDataset class
├── loaders.py                      # DataLoader builders
└── preprocessing.py                # Text preprocessing utilities
```

### Testing

```
tests/
├── unit/
│   ├── __init__.py
│   └── test_tokenizers.py          # Tokenizer unit tests
├── fixtures/
│   └── tokenizer_vocab.json        # Test tokenizer vocabulary
└── conftest.py                     # (update with dataset fixtures)
```

---

## Detailed Tasks

### Task 2.1: Enhance BaseTokenizer (from Phase 1)

**Objective:** Refine abstract tokenizer for concrete implementations

**File:** `src/fundamentallm/data/tokenizers/base.py`

**Current Status:** Created in Phase 1 with basic methods

**Enhancements Needed:**
- Add `batch_encode()` method for encoding multiple texts
- Add `batch_decode()` method for decoding multiple token sequences
- Add properties for special token IDs (pad, unk, bos, eos)
- Add validation for tokenizer state (must be trained before encode)
- Add type hints for all methods
- Consider edge cases (empty texts, None inputs)

**Success Criteria:**
- All abstract methods properly documented
- Batch methods have default implementations
- Can be imported and subclassed in Phase 2.2

---

### Task 2.2: Character Tokenizer Implementation

**Objective:** Create CharacterTokenizer for character-level language modeling

**File:** `src/fundamentallm/data/tokenizers/character.py`

**Key Features:**
1. **Special Tokens:**
   - `<PAD>` - Padding token
   - `<UNK>` - Unknown character
   - `<BOS>` - Beginning of sequence
   - `<EOS>` - End of sequence

2. **Vocabulary Building:**
   - Build from training texts
   - Min frequency filtering (optional)
   - Store char_to_id and id_to_char mappings

3. **Core Methods:**
   - `train(texts: List[str]) -> None`
     - Count character frequencies
     - Build vocab with special tokens
     - Initialize mappings
   
   - `encode(text: str, add_special_tokens: bool = False) -> List[int]`
     - Convert text to token IDs
     - Handle unknown characters → UNK token
     - Optionally add BOS/EOS tokens
   
   - `decode(tokens: List[int], skip_special_tokens: bool = True) -> str`
     - Convert token IDs back to text
     - Optionally skip special tokens
     - Handle invalid token IDs
   
   - `save(path: Path) -> None`
     - Serialize to JSON with metadata
     - Save char_to_id, id_to_char mappings
     - Save min_frequency setting
   
   - `load(path: Path) -> CharacterTokenizer` (classmethod)
     - Deserialize from JSON
     - Validate loaded state
     - Mark as trained

4. **Properties:**
   - `vocab_size: int` - Total vocabulary size
   - `pad_token_id: int` - ID of padding token
   - `unk_token_id: int` - ID of unknown token

**Implementation Notes:**
- Use JSON for serialization (human-readable)
- Ensure vocab_size includes special tokens
- Handle edge case: text with only unknown characters
- Make special tokens configurable for future extensions
- Use deterministic ordering for reproducibility

**Success Criteria:**
- ✅ Can train tokenizer on texts
- ✅ Can encode and decode with roundtrip
- ✅ Unknown characters handled correctly
- ✅ Special tokens work as expected
- ✅ Can save and load tokenizer
- ✅ Vocabulary size includes special tokens

---

### Task 2.3: Language Model Dataset Class

**Objective:** Create PyTorch Dataset for autoregressive language modeling

**File:** `src/fundamentallm/data/dataset.py`

**Class: LanguageModelDataset**

**Objective:** Map token sequences to (input, target) pairs for next-token prediction

**Constructor:**
```python
def __init__(
    self,
    token_ids: torch.Tensor,      # All tokens as 1D tensor
    sequence_length: int,          # Length of each sequence
    stride: Optional[int] = None   # Stride for overlapping (default: sequence_length)
)
```

**Key Methods:**

1. `__len__() -> int`
   - Calculate number of sequences
   - Formula: `max(0, (len(tokens) - seq_len - 1) // stride + 1)`
   - Reason: We need seq_len + 1 tokens for (input, target) pairs

2. `__getitem__(idx: int) -> Tuple[Tensor, Tensor]`
   - Return (input_ids, target_ids) pair
   - Input: tokens[start:start+seq_len]
   - Target: tokens[start+1:start+seq_len+1] (shifted by 1)
   - Enable next-token prediction

**Detailed Logic:**
```
Example with sequence_length=3:
All tokens: [1, 2, 3, 4, 5, 6, 7]

idx=0: input=[1,2,3], target=[2,3,4]
idx=1: input=[2,3,4], target=[3,4,5]   (stride=1 for overlapping)
idx=2: input=[3,4,5], target=[4,5,6]
idx=3: input=[4,5,6], target=[5,6,7]
```

**Configuration Options:**
- `stride=sequence_length`: Non-overlapping sequences (default)
- `stride=1`: Overlapping sequences (more data but redundant)

**Edge Cases to Handle:**
- Empty token list
- Tokens shorter than sequence_length
- Very long sequences (use stride to manage)

**Success Criteria:**
- ✅ Correct number of sequences calculated
- ✅ Indexing returns correct (input, target) pairs
- ✅ Can be used with torch DataLoader
- ✅ Handles edge cases without errors

---

### Task 2.4: DataLoader Builders

**Objective:** Create helper functions to build dataloaders with proper train/val split

**File:** `src/fundamentallm/data/loaders.py`

**Function: `create_dataloaders()`**

**Purpose:** Build train and validation dataloaders from raw text

**Parameters:**
```python
def create_dataloaders(
    text: str,                          # Raw text content
    tokenizer: BaseTokenizer,           # Trained tokenizer
    config: TrainingConfig,             # Training configuration
    return_tokenizer: bool = False      # Optional: return tokenizer
) -> Union[
    Tuple[DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, BaseTokenizer]
]
```

**Implementation Steps:**

1. **Tokenize entire text**
   - Call `tokenizer.encode(text)`
   - Convert to torch.Tensor of shape (num_tokens,)

2. **Token-level train/val split** (CRITICAL: Split at token level, not character level!)
   - Calculate split index: `train_size = int(len(tokens) * train_split)`
   - Split: `train_tokens = tokens[:train_size]`, `val_tokens = tokens[train_size:]`
   - Avoids data leakage (characters from val_tokens won't appear in training)

3. **Create datasets**
   - `LanguageModelDataset(train_tokens, sequence_length)`
   - `LanguageModelDataset(val_tokens, sequence_length)`

4. **Create dataloaders**
   - Train loader: `shuffle=True`, `drop_last=True` (for stable batch sizes)
   - Val loader: `shuffle=False`, `drop_last=False` (use all validation data)
   - Both: `pin_memory=True` (GPU optimization), `num_workers=config.num_workers`

**Pseudo-code:**
```
tokenize_text → [tokens]
split_tokens → [train_tokens, val_tokens]
create_datasets → [train_dataset, val_dataset]
create_dataloaders → [train_loader, val_loader]
```

**Configuration from TrainingConfig:**
- `sequence_length` - Length of sequences
- `batch_size` - Batch size
- `train_split` - Train/val ratio (default 0.9)
- `num_workers` - DataLoader workers

**Success Criteria:**
- ✅ Returns tuple of (train_loader, val_loader)
- ✅ Train/val split is at token level
- ✅ No overlap between train and validation tokens
- ✅ Can iterate over dataloaders
- ✅ Batches have correct shape

---

### Task 2.5: Data Preprocessing Utilities

**Objective:** Create text preprocessing functions

**File:** `src/fundamentallm/data/preprocessing.py`

**Functions to Implement:**

1. `load_text(path: Path, encoding: str = "utf-8") -> str`
   - Load text from file
   - Handle encoding errors gracefully
   - Strip leading/trailing whitespace

2. `clean_text(text: str) -> str`
   - Remove control characters
   - Normalize whitespace
   - Handle special cases (optional)
   - Config: aggressive cleaning vs minimal

3. `prepare_training_data(raw_text_path: Path, output_path: Path, clean: bool = True) -> str`
   - Load raw text
   - Apply preprocessing
   - Save to output
   - Return loaded text

**Simple Implementation Initially:**
- Focus on loading and basic cleaning
- Can be extended in later phases
- Avoid over-engineering

**Success Criteria:**
- ✅ Can load text files
- ✅ Handles encoding errors
- ✅ Cleans unwanted characters
- ✅ Works with various file sizes

---

### Task 2.6: Tokenizer Unit Tests

**Objective:** Comprehensive test coverage for tokenizer

**File:** `tests/unit/test_tokenizers.py`

**Test Class: TestCharacterTokenizer**

**Test Methods:**

1. `test_vocab_size()`
   - ✅ vocab_size > 4 (at least special tokens)
   - ✅ vocab_size includes all unique characters

2. `test_encode_decode_roundtrip()`
   - ✅ `tokenizer.encode(text) → tokens`
   - ✅ `tokenizer.decode(tokens) → text`
   - ✅ text == original for ASCII text

3. `test_unknown_character_handling()`
   - ✅ Character not in training data → UNK token
   - ✅ Decode UNK token → "<UNK>" or placeholder

4. `test_special_tokens()`
   - ✅ encode(text, add_special_tokens=True)
   - ✅ First token is BOS_TOKEN
   - ✅ Last token is EOS_TOKEN

5. `test_batch_encode_decode()`
   - ✅ batch_encode() works for multiple texts
   - ✅ batch_decode() works for token sequences
   - ✅ Results match individual encode/decode

6. `test_save_load()`
   - ✅ Save tokenizer to JSON
   - ✅ Load tokenizer from JSON
   - ✅ Loaded tokenizer produces same results
   - ✅ vocab_size matches after load

7. `test_padding()`
   - ✅ pad_token_id is valid
   - ✅ Padding doesn't corrupt other tokens

8. `test_edge_cases()`
   - ✅ Empty text handled
   - ✅ Very long text handled
   - ✅ Text with only unknown chars handled
   - ✅ Unicode characters (if applicable)

**Fixtures:**
- `tokenizer` - Pre-trained tokenizer instance
- `sample_texts` - List of test texts (from conftest.py)

**Success Criteria:**
- ✅ All tests pass
- ✅ Coverage > 90% for tokenizer module
- ✅ Edge cases covered

---

### Task 2.7: Dataset and DataLoader Tests

**Objective:** Test dataset and dataloader functionality

**File:** `tests/unit/test_data.py` (create new)

**Test Class: TestLanguageModelDataset**

**Test Methods:**

1. `test_dataset_length()`
   - ✅ Correct number of sequences
   - ✅ Length matches formula

2. `test_getitem_shape()`
   - ✅ Returns tuple of (input, target)
   - ✅ Both have correct shape: [sequence_length]

3. `test_getitem_values()`
   - ✅ Input and target correctly shifted
   - ✅ No overlaps or gaps

4. `test_stride_behavior()`
   - ✅ stride=sequence_length → non-overlapping
   - ✅ stride=1 → maximum overlap

5. `test_edge_cases()`
   - ✅ Empty tensor handled
   - ✅ Short sequences handled
   - ✅ Very long sequences handled

**Test Class: TestDataLoaders**

**Test Methods:**

1. `test_create_dataloaders()`
   - ✅ Returns (train_loader, val_loader)
   - ✅ Both are DataLoader instances

2. `test_train_val_split()`
   - ✅ Correct train/val ratio
   - ✅ No overlap between train and val tokens
   - ✅ All tokens accounted for

3. `test_batch_shape()`
   - ✅ Train batches: [batch_size, sequence_length]
   - ✅ Val batches: [batch_size, sequence_length]

4. `test_dataloader_iteration()`
   - ✅ Can iterate through train_loader
   - ✅ Can iterate through val_loader
   - ✅ Correct number of batches

**Fixtures:**
- `sample_tokenizer` - Trained tokenizer
- `sample_tokens` - Token tensor
- `train_config` - TrainingConfig instance

**Success Criteria:**
- ✅ All tests pass
- ✅ Coverage > 85% for data module

---

### Task 2.8: Integration Test - Tokenize and Load

**Objective:** End-to-end test of data pipeline

**File:** `tests/integration/test_data_pipeline.py` (create new)

**Test Class: TestDataPipeline**

**Test Method: test_end_to_end_data_pipeline()**

Steps:
1. Create sample text
2. Train tokenizer on text
3. Encode text to tokens
4. Create dataloaders
5. Iterate through one batch
6. Decode batch back to text
7. Verify quality

Expected Behavior:
- ✅ No errors at any step
- ✅ Batch shape correct
- ✅ Tokens valid (within vocab_size range)
- ✅ Decoded text readable

**Success Criteria:**
- ✅ Full pipeline works end-to-end
- ✅ No data corruption

---

### Task 2.9: Test Fixtures and Sample Data

**Objective:** Create test data and fixtures for consistent testing

**File:** `tests/conftest.py` (update from Phase 1)

**Add Fixtures:**

```python
@pytest.fixture
def sample_tokenizer():
    """Create and train tokenizer on sample data."""
    from fundamentallm.data.tokenizers.character import CharacterTokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(["hello world", "test data", "fundamentallm"])
    return tokenizer

@pytest.fixture
def sample_tokens(sample_tokenizer, sample_text):
    """Tokenize sample text."""
    import torch
    tokens = sample_tokenizer.encode(sample_text)
    return torch.tensor(tokens, dtype=torch.long)

@pytest.fixture
def train_config(tmp_path):
    """Create training config for testing."""
    from fundamentallm.config.training import TrainingConfig
    return TrainingConfig(
        data_path=tmp_path / "data.txt",
        batch_size=8,
        max_epochs=1,
        sequence_length=32,
        checkpoint_dir=tmp_path / "checkpoints"
    )
```

**File:** `tests/fixtures/sample_data.txt` (from Phase 1, verify content)

Ensure it has:
- Multiple paragraphs
- Variety of characters
- Sufficient length (1000+ chars)
- No special binary data

**Success Criteria:**
- ✅ All fixtures are accessible in tests
- ✅ Fixtures produce consistent results

---

## Implementation Checklist

- [ ] Enhance BaseTokenizer with batch methods (Task 2.1)
- [ ] Implement CharacterTokenizer class (Task 2.2)
  - [ ] Implement __init__
  - [ ] Implement train()
  - [ ] Implement encode()
  - [ ] Implement decode()
  - [ ] Implement save()
  - [ ] Implement load()
  - [ ] Implement vocab_size property
- [ ] Implement LanguageModelDataset class (Task 2.3)
  - [ ] Implement __init__
  - [ ] Implement __len__()
  - [ ] Implement __getitem__()
- [ ] Implement create_dataloaders function (Task 2.4)
- [ ] Implement preprocessing utilities (Task 2.5)
- [ ] Create tokenizer tests (Task 2.6)
- [ ] Create dataset and dataloader tests (Task 2.7)
- [ ] Create integration tests (Task 2.8)
- [ ] Update conftest.py with fixtures (Task 2.9)
- [ ] Verify all tests pass
- [ ] Check test coverage > 85%

---

## Success Criteria for Phase 2

1. **Tokenizer Implementation**
   - ✅ CharacterTokenizer can train on texts
   - ✅ Encoding/decoding works with roundtrip
   - ✅ Special tokens handled correctly
   - ✅ Can save and load from disk
   - ✅ All edge cases handled

2. **Dataset Implementation**
   - ✅ LanguageModelDataset creates correct (input, target) pairs
   - ✅ Supports configurable stride
   - ✅ Works with torch DataLoader

3. **DataLoader Builders**
   - ✅ create_dataloaders returns valid loaders
   - ✅ Train/val split is at token level (no leakage)
   - ✅ Can iterate through batches

4. **Preprocessing**
   - ✅ Can load text from files
   - ✅ Can clean/preprocess text
   - ✅ Handles various encodings

5. **Testing**
   - ✅ Unit tests for tokenizer
   - ✅ Unit tests for dataset
   - ✅ Integration tests for full pipeline
   - ✅ Coverage > 85%
   - ✅ All tests pass

6. **Documentation**
   - ✅ Docstrings for all classes/functions
   - ✅ Type hints complete
   - ✅ Usage examples in docstrings

---

## Next Phase Dependency

Phase 2 must be complete before starting Phase 3 (Model Architecture).

Phase 3 will:
- Use TrainingConfig from Phase 1
- Import DataLoader from Phase 2
- Train model on data from Phase 2

---

## Critical Notes

### Data Leakage Prevention
- ⚠️ CRITICAL: Split at token level, NOT character level
- ⚠️ CRITICAL: No overlap between train and validation
- ⚠️ CRITICAL: Test this explicitly in tests

### Tokenizer Design
- Minimize overhead in train/encode/decode (called frequently)
- JSON serialization is fine for educational project
- Special tokens pattern (BOS/EOS/PAD) is standard in NLP

### DataLoader Best Practices
- `pin_memory=True` only with GPU training
- `drop_last=True` in training for stable batch statistics
- `num_workers` depends on data loading speed (start with 4)
- `shuffle=True` only for training

### Testing Strategy
- Focus on edge cases (empty data, very short sequences)
- Test both happy path and error conditions
- Use fixtures for consistency
- Aim for 85%+ coverage

---

## Extension Points (Future Phases)

Phase 2 is designed to support:
- Multiple tokenizer types (BPE in future)
- Custom preprocessing pipelines
- Different dataset implementations (sliding window, non-overlapping)
- Different sampling strategies
