# Embedding Bag

## Overview
Coalesced embedding lookups with aggregation for efficient vocabulary retrieval.

## Why We Built It
- **LLM input**: First operation in transformer - lookup token embeddings  
- **Coalesced reads**: Group nearby indices for better memory bandwidth
- **Aggregation**: Support for mean/sum/max pooling

## Performance

| Config | Metal | PyTorch | Status |
|--------|-------|---------|--------|
| 32K vocab, 4096 dim | ~0.1ms | ~0.1ms | 🔵 Close |

## Usage
```python
from metalcore import embedding_bag

embeddings = embedding_bag(weight, indices, offsets, mode='sum')
```

## Notes
- Standard embedding lookup already efficient on MPS
- Bag aggregation useful for variable-length sequences
- Supports multi-table lookup for hybrid vocabularies
