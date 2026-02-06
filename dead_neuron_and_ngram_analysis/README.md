# Neuron Analysis for Large Language Models

This project provides comprehensive analysis tools for studying neuron behavior in large language models, particularly Mamba and GPT-2 architectures. It implements analysis of dead neurons, positional neurons, delta-sensitive neurons, attention neurons, and n-gram triggers, with ablation studies and perplexity evaluation.

## Overview

The project analyzes different types of neurons in transformer and state-space models:
- **Dead neurons**: Neurons that rarely or never activate
- **Positional neurons**: Neurons that respond to specific token positions
- **Delta-sensitive neurons**: Neurons that respond to changes in input
- **Attention neurons**: Neurons derived from attention mechanisms
- **N-gram neurons**: Neurons triggered by specific n-gram patterns

## Installation

### Requirements

```bash
pip install torch transformers datasets matplotlib seaborn numpy
```

### Dependencies

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- NumPy
- Matplotlib
- Seaborn

## Running the Code

### 1. Comprehensive Ablation Study

Run the comprehensive ablation study that analyzes dead, positional, and delta-sensitive neurons with perplexity evaluation:

```bash
python3 run_comprehensive_ablation.py
```

This script:
- Loads a Mamba model (default: `state-spaces/mamba-130m-hf`)
- Loads WikiText-2 dataset
- Runs comprehensive analysis including:
  - Dead neuron identification
  - Positional neuron identification
  - Delta-sensitive neuron identification
  - Attention neurons analysis
  - Ablation studies with perplexity evaluation
- Generates visualizations and saves them to `analysis_outputs/plots/`

**Output:**
- Ablation results showing perplexity changes after pruning different neuron types
- Comprehensive visualizations of neuron analysis
- Results summary printed to console

### 2. N-gram Analysis

Run n-gram analysis to reproduce Figures 2-4 from the paper "Neurons in Large Language Models: Dead, N-gram, Positional":

```bash
python3 run_ngram_analysis.py
```

This script:
- Analyzes multiple models (GPT-2, Mamba-130M, Mamba-370M, Mamba-790M, Mamba-1.4B, Mamba-2.8B)
- Identifies n-gram triggers for neurons (unigrams, bigrams, trigrams)
- Generates paper-style figures:
  - **Figure 2**: Stacked histograms of neurons categorized by unigram triggers (per model)
  - **Figure 3a**: Token-detecting neurons per layer (overlay across models)
  - **Figure 3b**: Token coverage per layer (overlay across models)
  - **Figure 4**: New tokens covered per layer (overlay across models)
- Saves results to log files for offline plotting

**Output:**
- Individual Figure 2 plots for each model in `plots/`
- Overlay plots (Figure 3a, 3b, 4) in `plots/`
- Analysis results logged to `logs/ngram_analysis_*.json`

### 3. Main Analysis Script

Run the main comprehensive analysis script with custom parameters:

```bash
python3 main.py --models state-spaces/mamba-130m-hf --layer 0 --top_k 10 --text_limit 300
```

**Arguments:**
- `--models`: List of model names to analyze (default: `state-spaces/mamba-130m-hf`)
- `--layer`: Layer index to analyze (default: 1)
- `--top_k`: Number of top delta-sensitive neurons (default: 10)
- `--text_limit`: Limit number of texts to process (default: None, uses all)
- `--save_results`: Save results to JSON file
- `--pruning_ratio`: Ratio of neurons to prune (default: 0.15)

**Output:**
- Comprehensive neuron analysis results
- Visualizations saved to `analysis_outputs/plots/`
- Optional JSON results file if `--save_results` is used

## Project Structure

```
.
├── main.py                          # Main analysis script with comprehensive neuron analysis
├── run_comprehensive_ablation.py   # Comprehensive ablation study script
├── run_ngram_analysis.py           # N-gram analysis script (Figures 2-4)
├── utils.py                         # Utility functions for model structure handling
├── neuron_characterization.py       # Dead and positional neuron identification
├── delta_extraction.py              # Delta-sensitive neuron analysis
├── attention_neurons.py             # Attention-based neuron analysis
├── visualization_module.py         # Visualization functions
└── README.md                        # This file
```

## Core Modules

### `main.py`
Main analysis script that orchestrates comprehensive neuron analysis:
- Dead neuron identification
- Positional neuron identification
- Delta-sensitive neuron analysis
- Attention neurons analysis
- Ablation studies with perplexity evaluation
- Comprehensive visualization generation

### `neuron_characterization.py`
Functions for identifying and analyzing different neuron types:
- `find_dead_neurons()`: Identifies neurons that rarely activate
- `find_positional_neurons()`: Identifies position-sensitive neurons
- `run_complete_neuron_analysis()`: Complete pipeline for neuron analysis
- `plot_neuron_activation_distribution()`: Visualizes activation distributions

### `delta_extraction.py`
Analysis of delta-sensitive neurons:
- `find_delta_sensitive_neurons_fixed()`: Identifies neurons sensitive to input changes
- `evaluate_perplexity()`: Evaluates model perplexity

### `attention_neurons.py`
Attention-based neuron analysis:
- `MambaAttentionNeurons`: Class for creating neurons from attention vectors
- `integrate_mamba_attention_neurons()`: Convenience function for attention analysis
- Multiple neuron creation methods: `attention_weighted`, `gradient_guided`, `rollout`

### `visualization_module.py`
Comprehensive visualization functions:
- `create_comprehensive_report()`: Generates all visualizations
- `plot_attention_neurons_analysis()`: 6-panel attention neuron visualization
- `plot_dead_neuron_stats_by_layer()`: Dead neuron statistics across layers
- Various other plotting functions

### `utils.py`
Utility functions:
- `get_model_layers()`: Extracts layer structure from models
- `get_activation_hook_target()`: Determines where to hook for activations
- `debug_model_structure()`: Debugging tool for model structure

## Attention Neurons Integration

### Overview

The `attention_neurons.py` module has been successfully integrated into the main project. It provides comprehensive attention-based neuron analysis for Mamba models.

### Features

- **MambaAttentionNeurons class**: Handles both direct `model.layers` and `model.backbone.layers` structures
- **Attention vector extraction**: Works with real Mamba models
- **Neuron creation methods**: Multiple approaches (attention_weighted, gradient_guided, rollout)
- **Neuron behavior analysis**: Comprehensive analysis of activation patterns
- **Integration function**: Convenience function for easy integration

### Usage

#### Basic Usage

```python
from attention_neurons import integrate_mamba_attention_neurons

# Run attention neurons analysis
results = integrate_mamba_attention_neurons(
    model, 
    sample_input, 
    layer_indices=[0, 1], 
    methods=['attention_weighted']
)
```

#### In Main Analysis

Attention neurons analysis is automatically included when running:

```python
from main import run_comprehensive_analysis

results = run_comprehensive_analysis(model, tokenizer, texts, layer_idx=0)

# Access attention neurons results
if 'attention_neurons' in results['analysis_results']:
    attention_results = results['analysis_results']['attention_neurons']
```

### How It Works

1. **Attention Vector Extraction**: Extracts attention matrices and vectors from model layers
2. **Neuron Creation**: Creates neurons using different methods (attention_weighted, gradient_guided, rollout)
3. **Behavior Analysis**: Analyzes activation patterns, importance, and diversity
4. **Visualization**: Generates 6-panel comprehensive visualizations

### Visualization Features

The `plot_attention_neurons_analysis()` function creates a 6-panel visualization:
- **Panel 1**: Neuron activation distribution histogram
- **Panel 2**: Top neurons by activation value
- **Panel 3**: Activation vs importance scatter plot
- **Panel 4**: Attention heatmap (if available)
- **Panel 5**: Neuron diversity score
- **Panel 6**: Statistical summary

### Technical Details

#### Model Structure Handling
- **Direct layers**: `model.layers` (for custom models)
- **Backbone layers**: `model.backbone.layers` (for HuggingFace Mamba models)
- **Automatic detection**: Automatically detects and uses the correct structure

#### Error Handling
- **Graceful degradation**: If attention analysis fails, main analysis continues
- **Fallback mechanisms**: Uses dummy data or skips problematic analyses
- **User feedback**: Clear success/failure messages with detailed information

#### Performance
- **Efficient processing**: Processes multiple layers simultaneously
- **Memory management**: Handles large attention matrices efficiently
- **GPU support**: Works with both CPU and GPU models

## Output Structure

### Directory Structure

After running analyses, the following structure is created:

```
analysis_outputs/
├── plots/                    # All generated visualizations
│   ├── *_ablation_comprehensive.png
│   ├── *_dead_neurons.png
│   ├── *_positional_neurons.png
│   ├── *_attention_neurons.png
│   └── ...
├── logs/                     # Analysis logs (for n-gram analysis)
│   └── ngram_analysis_*.json
└── data/                     # Structured data (if saved)
    └── neuron_analysis_results_*.json
```

### Generated Visualizations

1. **Ablation Study Plots**: Perplexity changes after pruning different neuron types
2. **Dead Neuron Statistics**: Distribution of dead neurons across layers
3. **Positional Neuron Scatter Plots**: Position vs correlation plots
4. **Attention Neuron Analysis**: 6-panel comprehensive visualizations
5. **N-gram Analysis Figures**: Paper-style Figures 2-4
6. **Activation Distributions**: Histograms and scatter plots of neuron activations

## Analysis Results

### Ablation Study Results

The ablation study evaluates the impact of pruning different neuron types:

- **Baseline perplexity**: Model performance without pruning
- **Dead neurons ablated**: Perplexity after removing dead neurons
- **Positional neurons ablated**: Perplexity after removing positional neurons
- **Delta-sensitive neurons ablated**: Perplexity after removing delta-sensitive neurons
- **Combined ablation**: Perplexity after removing dead + positional neurons

### Neuron Characterization

The analysis identifies:
- **Dead neurons**: Count and distribution across layers
- **Positional neurons**: Count and position correlations
- **Delta-sensitive neurons**: Top-k most sensitive neurons
- **Attention neurons**: Activation patterns and importance scores
- **Overlap analysis**: Intersections between different neuron types

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure you have internet connection for downloading models
2. **CUDA out of memory**: Reduce `--text_limit` or use smaller models
3. **Missing dependencies**: Install all required packages from requirements
4. **Import errors**: Ensure all Python files are in the same directory

### Log Analysis

For n-gram analysis, check log files:
```bash
ls -la logs/
cat logs/ngram_analysis_*.json
```

### Data Validation

Validate JSON results:
```python
import json

with open('analysis_outputs/data/neuron_analysis_results_*.json', 'r') as f:
    data = json.load(f)

print(f"Models analyzed: {data.keys()}")
```
