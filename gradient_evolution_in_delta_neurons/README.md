# Delta Reasoning Analysis

This repository contains analysis tools for investigating perplexity changes and gradient amplification in language models, with a focus on comparing Mamba and Transformer architectures.

## Overview

The project analyzes how different neural network architectures (specifically Mamba and GPT-2) respond to neuron perturbations and how these perturbations affect model performance. The analysis investigates why certain architectures show higher percentage changes in perplexity when specific neurons are perturbed.

## Main Analysis Scripts

### Delta Percentage Analysis

The delta percentage analysis script investigates why Mamba models show high percentage changes in perplexity when delta-sensitive neurons are perturbed. This analysis:

- Compares baseline and perturbed perplexity values for both Mamba and GPT-2 models
- Analyzes the relationship between baseline perplexity and percentage sensitivity
- Examines absolute versus percentage changes across different factual relations
- Identifies root causes of high delta percentages, including the denominator effect from lower baseline perplexity values

The analysis focuses on multiple factual relations including capital cities, original networks, diplomatic relations, organizational memberships, and record labels, among others.

### Jacobian Analysis

The Jacobian analysis script measures gradient amplification and sensitivity in language models. This comprehensive analysis:

- Computes gradient sensitivity using perturbation-based approaches
- Estimates Lipschitz constants to measure maximum amplification factors
- Calculates output variance across different inputs
- Determines effective rank of hidden state representations
- Performs layer-wise analysis across all model layers

The analysis compares Mamba and GPT-2 models across multiple metrics to understand how each architecture amplifies or dampens gradients through the network.

## Supporting Modules

### Delta Extraction

Provides functions for:
- Extracting delta parameters from specific model layers
- Finding neurons sensitive to delta computation
- Registering perturbation hooks for neuron manipulation
- Evaluating perplexity changes before and after perturbations

### Factual Recall

Contains utilities for:
- Loading and managing relation-specific prompts for factual knowledge evaluation
- Setting up models and tokenizers
- Running factual recall analysis with relation-specific perplexity evaluation
- Generating comparison tables and analysis reports

### Utilities

Provides helper functions for:
- Debugging model structures
- Extracting model layers from different architectures
- Handling various model configurations (Mamba, Transformer, etc.)

## Dependencies

The project requires:
- PyTorch for model operations and tensor computations
- Transformers library for loading pre-trained models
- NumPy for numerical computations
- Pandas for data manipulation and analysis
- Matplotlib and Seaborn for visualization
- Datasets library for loading text datasets

## Usage

### Running Delta Percentage Analysis

Execute the delta percentage analysis to investigate perplexity change patterns:

```bash
python delta_percentage_analysis.py
```

or

```bash
python3 delta_percentage_analysis.py
```

The script automatically:
- Loads Mamba and GPT-2 models
- Processes multiple factual relations
- Computes baseline and perturbed perplexity values
- Generates visualizations comparing absolute and percentage changes
- Saves results to CSV format

### Running Jacobian Analysis

Execute the Jacobian analysis with optional parameters:

**Run analysis on all layers (default):**
```bash
python jacobian_analysis.py
```

or

```bash
python3 jacobian_analysis.py
```

**Run analysis on a specific layer:**
```bash
python jacobian_analysis.py --layer 1
```

**Run analysis with custom number of texts:**
```bash
python jacobian_analysis.py --num_texts 50
```

**Run analysis on a specific layer with custom number of texts:**
```bash
python jacobian_analysis.py --layer 1 --num_texts 5
```

**Command-line arguments:**
- `--layer`: Layer index to analyze (default: None = all layers)
- `--num_texts`: Number of texts to use for evaluation (default: 30)

The script outputs comprehensive metrics including gradient sensitivity, Lipschitz constants, effective rank, and output variance for each layer in both models.

## Key Insights

The analyses reveal several important findings:

1. **Baseline Perplexity Effect**: Mamba models often have lower baseline perplexity values, which can lead to higher percentage changes when the same absolute perturbation is applied.

2. **Gradient Amplification**: The Jacobian analysis helps identify whether Mamba or Transformer architectures exhibit different gradient amplification patterns through their layers.

3. **Layer-wise Variations**: Different layers show varying sensitivity to perturbations, with some layers being more critical for maintaining factual knowledge.

4. **Architecture Differences**: The comparison between Mamba and Transformer architectures reveals fundamental differences in how they process and represent information.

## Output Files

The analyses generate various output files:
- CSV files containing detailed analysis results
- JSON files with comprehensive metrics and statistics
- Visualization images comparing models and metrics
- Summary tables printed to console

## Notes

- The analyses require GPU support for efficient computation, though CPU execution is possible
- Model loading may take time depending on network speed and available resources
- Some analyses process multiple layers and can take significant computation time
- Results are cached where possible to avoid redundant computations
