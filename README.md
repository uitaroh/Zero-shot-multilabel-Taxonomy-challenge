#My first attempt at zero-shot multilabel learning on a raw dataset

## Project Overview
This project develops a scalable **company classifier** for an insurance taxonomy using unstructured data (descriptions, tags, and sector labels). Without ground-truth labels, the solution emphasizes **heuristic validation** over traditional metrics, leveraging embeddings and zero-shot learning while prioritizing **real-world relevance**.

**Key Deliverables:**
- Annotated dataset
- Modular code implementation
- Discussion of trade-offs between precision and scalability

## Project Structure
```
veridion-challenge/
├── data/
│ ├── clean_data.csv # Cleaned company dataset
│ └── clean_labels+description.csv # Processed taxonomy labels with descriptions
├── notebooks/
│ ├── Veridion_Challenge.preprocessing.ipynb # Data cleaning notebook
│ ├── Veridion_challenge_labels.preprocessing.ipynb # Label processing notebook
│ └── Main_Veridion_Challenge.ipynb # Main modeling notebook
└── README.md
```

## Methodology

### 1. Data Preparation & Preprocessing
#### Dataset Cleaning
- **Text Normalization**: Lowercasing, whitespace handling, non-string input safety
- **Duplicate Handling**: Semantic duplicate detection with manual verification
- **Missing Values**: Hybrid approach using:
  - FastText classification with Optuna tuning (automated)
  - Manual imputation based on business context

#### Label Processing
- Standardized insurance taxonomy labels through:
  - Case normalization and special character removal
  - Cosine similarity-based duplicate detection (threshold=0.65)
  - LLM-generated descriptions for each label (ChatGPT & DeepSeek)

### 2. Modeling Approaches
#### Zero-Shot Classification
- **Hybrid Model**: SentenceTransformer embeddings + BART-large-MNLI
  - Two complementary approaches:
    1. **Company-centered**: Predict labels for companies
    2. **Taxonomy-centered**: Retrieve companies matching each label
- **ConsensusInsuranceClassifier**:
  - Combines predictions from both models
  - Weighted scoring (30% SentenceTransformer + 70% BART)
  - Thresholds: ST=0.7, BART=0.6

#### Few-Shot Classification
*(Details to be added)*

## Key Findings
- The zero-shot approach achieved strong precision but limited coverage
- Rare labels benefited from the inverted taxonomy-centered approach
- LLM-generated label descriptions significantly improved model performance

## Implementation Notes
- **Embedding Model**: all-MiniLM-L6-v2
- **Rejected Model**: ROBERTA (too slow for production use)
- **Computational Trade-offs**:
  - Lemmatization/stopword removal omitted for large-scale deployment
  - BART chosen for better speed/accuracy balance

## Future Improvements
1. Expand few-shot learning capabilities
2. Optimize for big data scenarios
3. Enhance handling of emerging categories in streaming data

---
