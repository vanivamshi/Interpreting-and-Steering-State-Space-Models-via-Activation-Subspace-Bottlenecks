# Mamba Neuron Analysis

## Overview
This project provides comprehensive tools for analyzing Mamba model neurons and their behaviors, including delta-sensitive neurons, causal neurons, attention neurons, and sparsity analysis.

## Features

### 1. Basic Mamba Analysis
- **Delta-Sensitive Neurons**: Identifies neurons sensitive to state changes
- **Causal Neurons**: Finds neurons with causal impact on model outputs
- **Inter-Layer Analysis**: Analyzes causal impact across layers
- **Cross-Layer Analysis**: Examines influence between different layers

### 2. Attention Neurons Analysis
Integrated attention neurons analysis for Mamba models with multiple methods:
- `attention_weighted`: Creates neurons weighted by attention vectors
- `gradient_guided`: Creates neurons guided by gradients (XAI vectors)
- `rollout`: Creates neurons using rollout attention method

### 3. Comprehensive Sparsity Analysis
Multiple sparsity detection methods for robust analysis:

#### Percentile-Based Sparsity (Most Reliable)
- Uses percentiles (5%, 10%, 25%, 50%, 75%, 90%, 95%) of activation variance
- No arbitrary thresholds - adapts to actual data distribution
- Shows sparsity at different levels of activation intensity

#### Entropy-Based Sparsity
- Normalizes activations to [0,1] range
- Calculates entropy of activation distribution
- Higher entropy = more uniform (less sparse)
- Lower entropy = more concentrated (more sparse)

#### Gini Coefficient Sparsity
- Measures inequality in activation distribution
- Gini = 0: Perfect equality (uniform activations)
- Gini = 1: Perfect inequality (very sparse)
- Higher Gini = more sparse

#### Optimal Threshold Sparsity
- Automatically finds threshold to achieve target sparsity
- Uses percentile-based approach
- Adapts to each model's activation characteristics

### 4. Model Efficiency Analysis
- Parameter counting
- Activation pattern analysis
- Sparsity calculations
- Efficiency metrics

### 5. Model Comparison
Comprehensive comparison between Mamba and Transformer models with:
- Layer dynamics analysis
- Sparsity comparison
- Efficiency metrics

## Sparsity Threshold Updates

### Model-Specific Thresholds
Updated sparsity calculations to use model-specific thresholds:
- **Mamba models**: 0.01 (higher threshold)
- **Transformer models**: 1e-5 (lower threshold)

### Why Different Thresholds?

**Mamba Models (0.01)**
- Mamba models have different activation patterns compared to Transformers
- They often have higher baseline activation values
- A higher threshold (0.01) is needed to properly identify "dead" neurons
- This prevents over-estimating sparsity due to architectural differences

**Transformer Models (1e-5)**
- Transformers typically have more sensitive activation patterns
- They can have very small but meaningful activations
- A lower threshold (1e-5) ensures we don't miss neurons that are actually active
- This provides more accurate sparsity measurements for Transformer architectures

### Impact
- Model-specific thresholds provide more accurate sparsity measurements
- Fair comparison between Mamba and Transformer architectures
- Better identification of truly "dead" vs. "active" neurons
- More reliable efficiency analysis and model comparison

## Multi-Layer Analysis

### Multi-Layer Comprehensive Sparsity
- Analyzes layers 0, 1, and 2 for comprehensive sparsity
- Better understanding of sparsity patterns across model depth
- Multiple sparsity detection methods for robust analysis

### Key Improvements
- **More Reliable Sparsity Values**: P50 percentile automatically adapts to activation scale
- **Multiple Methods Comparison**: Traditional + Entropy + Gini + Optimal + Percentile
- **Multi-Layer Analysis**: Layers 0, 1, 2 comprehensive analysis
- **Enhanced Visualizations**: Multiple plot types with different views

## Usage

### Basic Usage
```bash
python main.py --enable_attention
```

### Custom Configuration
```bash
python main.py \
  --model "state-spaces/mamba-130m-hf" \
  --layer 2 \
  --enable_attention \
  --attention_methods attention_weighted gradient_guided \
  --text_limit 200 \
  --save_results \
  --plots_dir "my_plots"
```

### Command-Line Arguments
- `--model`: Model to analyze (default: "state-spaces/mamba-130m-hf")
- `--layer`: Layer index to analyze (default: 0)
- `--top_k`: Top K neurons to analyze (default: 10)
- `--text_limit`: Limit number of texts to process (default: 100)
- `--enable_attention`: Enable attention neurons analysis
- `--attention_methods`: Methods for attention analysis (default: attention_weighted, gradient_guided, rollout)
- `--save_results`: Save analysis results to JSON file
- `--plots_dir`: Directory to save plots (default: plots)

## Analysis Pipeline

The main analysis pipeline includes:
1. **Basic Mamba analysis** - Delta-sensitive and causal neurons
2. **Inter-layer analysis** - Causal impact across layers
3. **Cross-layer analysis** - Influence between layers
4. **Attention neurons analysis** - Multiple attention methods
5. **Efficiency analysis** - Model efficiency metrics
6. **Comparison analysis** - Mamba vs Transformer comparison

## Output

### Results Storage
- Analysis results stored in `analysis_results` dictionary
- Can be saved to JSON files using `--save_results` flag
- Results saved to `analysis_outputs/` directory

### Key Metrics
- Total parameters
- Active neurons
- Efficiency ratio
- Activation variance
- Sparsity (multiple methods)
- Attention neuron metrics (neurons analyzed, mean activation, neuron diversity)

## Dependencies

- `torch`: PyTorch for deep learning operations
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `transformers`: Hugging Face transformers library
- `datasets`: Hugging Face datasets library

## Files

### Main Files
- `main.py`: Main script for running comprehensive analysis
- `utils.py`: Utility functions for model structure debugging
- `delta_extraction.py`: Delta-sensitive neuron extraction
- `causal_analysis.py`: Causal neuron analysis
- `attention_neurons.py`: Attention neurons integration
- `comparison_wrapper.py`: Model comparison wrapper
- `comparison_plots.py`: Comparison plotting functionality

## Recommendations

### For Model Comparison
1. Start with percentile-based sparsity - most reliable across models
2. Compare P50 values - median sparsity is most representative
3. Use entropy-based for pattern analysis
4. Use Gini-based for inequality comparison

### For Single Model Analysis
1. Use comprehensive analysis to get all methods
2. Identify most reasonable values (closest to 0.5)
3. Focus on that method for further analysis
4. Use optimal threshold for specific sparsity targets

## Summary

This project provides:
- ✅ Robust sparsity detection across different model architectures
- ✅ Multiple perspectives on sparsity (percentile, entropy, Gini)
- ✅ Automatic adaptation to different activation scales
- ✅ Comprehensive neuron analysis (delta, causal, attention)
- ✅ Model efficiency and comparison tools
- ✅ No arbitrary thresholds that fail on different models

The analysis tools solve fundamental problems of traditional sparsity detection and provide reliable and informative analysis of neural network activation patterns.

