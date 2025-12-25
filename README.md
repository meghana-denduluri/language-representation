# Language Representation

A comprehensive implementation of word embeddings using co-occurrence matrix factorization on the OpenWebText dataset. This project builds word representations from scratch using Singular Value Decomposition (SVD) on word co-occurrence statistics.

## Overview

This project implements a complete pipeline for learning distributed word representations:

1. **Data Processing**: Tokenization and preprocessing of the OpenWebText sentences dataset
2. **Vocabulary Construction**: Building vocabulary with frequency-based filtering
3. **Co-occurrence Matrix**: Computing word co-occurrence statistics with configurable context windows
4. **Dimensionality Reduction**: Applying SVD to generate dense word embeddings
5. **Evaluation**: Analyzing embeddings through similarity tasks and visualizations

## Directory Structure

```
language-representation/
├── language_representation.ipynb    # Main Jupyter notebook with complete implementation
├── Report.pdf                       # Detailed project report
├── concurrence_data/                # Co-occurrence matrix and vocabulary data
│   ├── cooccurrence_matrix.npz     # Sparse co-occurrence matrix
│   ├── vocabulary.pkl              # Vocabulary dictionary
│   ├── word_to_idx.json           # Word to index mapping
│   ├── idx_to_word.json           # Index to word mapping
│   ├── vocab_metadata.json        # Vocabulary statistics
│   └── cooccur_metadata.json      # Co-occurrence statistics
├── embeddings_data/                # Generated word embeddings
│   ├── embeddings_d50.npy         # 50-dimensional embeddings
│   ├── embeddings_d100.npy        # 100-dimensional embeddings
│   ├── embeddings_d200.npy        # 200-dimensional embeddings
│   ├── embedding_metadata_d*.json # Embedding metadata
│   ├── word_to_idx_d*.json        # Word mappings per dimension
│   └── idx_to_word_d*.json        # Index mappings per dimension
└── window_size_experiments/        # Window size ablation studies
    ├── window_2/                   # Context window = 2
    ├── window_5/                   # Context window = 5
    ├── window_10/                  # Context window = 10
    ├── window_15/                  # Context window = 15
    └── experiment_summary.json     # Experiment results summary
```

## Dependencies

The project requires the following Python libraries:

### Core Libraries
- `datasets` - HuggingFace datasets for loading OpenWebText
- `numpy` - Numerical computing and array operations
- `scipy` - Sparse matrix operations and scientific computing
- `scikit-learn` - SVD (TruncatedSVD) and t-SNE for dimensionality reduction

### NLP & Embeddings
- `gensim` - Word2Vec and pre-trained embeddings (Word2Vec, GloVe)

### Visualization & Analysis
- `matplotlib` - Plotting and visualizations
- `pandas` - Data manipulation and analysis

### Utilities
- `pickle` - Serialization
- `json` - JSON file handling
- `re` - Regular expressions for text processing

### Installation

```bash
pip install datasets numpy scipy scikit-learn gensim matplotlib pandas
```

## Running the Project

### Option 1: Jupyter Notebook (Recommended)

1. **Open the notebook**:
   ```bash
   jupyter notebook language_representation.ipynb
   ```

2. **Run cells sequentially**:
   - The notebook is organized into sections (0-7)
   - Each section builds upon the previous one
   - Cell outputs are preserved for reference

### Option 2: Google Colab

The notebook is designed to run on Google Colab with Google Drive integration:

1. Upload the notebook to Google Colab
2. Mount Google Drive for caching datasets
3. Run all cells in order

### Key Configuration Parameters

**Tokenization** (`TOKENIZATION_CONFIG`):
- `lowercase`: True
- `keep_numbers`: True
- `keep_punctuation`: False
- `min_token_length`: 1

**Vocabulary** (`VOCAB_CONFIG`):
- `subset_size_small`: 50,000 sentences
- `subset_size_large`: 300,000 sentences
- `min_frequency`: 5 (minimum word occurrences)
- `unk_token`: '<UNK>'

**Co-occurrence** (`COOCCURRENCE_CONFIG`):
- `window_size`: 5 (default context window)
- `use_symmetric`: True (symmetric co-occurrence counting)

**Embeddings** (`EMBEDDING_CONFIG`):
- Dimensions tested: 50, 100, 200, 300
- Algorithm: Truncated SVD

## Approach

### 1. Data Loading & Preprocessing

- **Dataset**: PaulPauls/openwebtext-sentences from HuggingFace (307M+ sentences)
- **Tokenization**: Custom regex-based tokenizer
  - Lowercase normalization
  - Whitespace normalization
  - Alphanumeric token extraction
  - Configurable number and punctuation handling

### 2. Vocabulary Construction

- Process 300,000 sentences from the dataset
- Apply frequency-based filtering (min_frequency = 5)
- Handle rare words with `<UNK>` token
- **Results**: 
  - 41,731 unique words in vocabulary
  - 6.6M total tokens processed
  - Average 22.12 tokens per sentence

### 3. Co-occurrence Matrix Building

- Sliding window approach with configurable window size
- Symmetric co-occurrence counting
- Sparse matrix representation for memory efficiency
- **Window size experiments**: Tested windows of 2, 5, 10, and 15

### 4. Dimensionality Reduction (SVD)

- Apply Truncated SVD to co-occurrence matrix
- Generate dense embeddings in multiple dimensions (50, 100, 200, 300)
- Preserve semantic relationships in lower-dimensional space

### 5. Evaluation & Analysis

- **Similarity Analysis**: Cosine similarity between word pairs
- **Visualization**: t-SNE for 2D projection of embeddings
- **Comparison**: Benchmark against pre-trained Word2Vec and GloVe
- **Ablation Studies**: Impact of window size and embedding dimensions 

## Dataset

**OpenWebText Sentences** (PaulPauls/openwebtext-sentences)
- Source: HuggingFace Datasets
- Size: 307,432,490 sentences
- Format: Single 'text' column with one sentence per row
- Domain: Web text from diverse sources

## Author

Meghana Denduluri

## License

This project is for educational and research purposes.
