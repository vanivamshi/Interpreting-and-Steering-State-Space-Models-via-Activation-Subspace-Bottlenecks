# Extended N-gram Analysis for Language Models

This repository contains analysis tools for studying n-gram patterns, head-like behavior, and receptive field properties in large language models, with a focus on Mamba and GPT-2 architectures.

## Overview

The main analysis script (`ngram_analysis.py`) performs comprehensive n-gram analysis across multiple model sizes, examining how neurons detect and respond to different n-gram patterns (1-grams, 2-grams, 3-grams) across model layers. The analysis is based on methodologies from "In-Context Language Learning: Architectures and Algorithms" and related research on neuron behavior in transformer and state-space models.

## Features

- **N-gram Trigger Collection**: Identifies which neurons activate in response to specific n-gram patterns
- **Head-like Behavior Analysis**: Groups neurons into head-like components and analyzes their specialization patterns
- **Receptive Field Analysis**: Measures how much each layer depends on the last n tokens
- **Linear Token Decode**: Tests whether token identity is linearly decodable from layer activations
- **Gradient Attribution**: Computes gradient importance for understanding neuron contributions
- **Kernel Interpolation Experiments**: Causal tests for Mamba models examining state-kernel effects
- **Multi-model Support**: Analyzes Mamba-130M, Mamba-370M, Mamba-790M, Mamba-1.4B, and GPT-2

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Datasets (Hugging Face)

## Installation

Install the required dependencies using pip. Ensure you have PyTorch installed with CUDA support if you plan to use GPU acceleration.

## Usage

Run the main analysis script directly. The script will:

1. Load models from Hugging Face (Mamba and GPT-2 variants)
2. Load text data from WikiText-2 dataset
3. Analyze specified layers for each model
4. Generate plots and save results to `plots/` and `logs/` directories

The analysis processes multiple models sequentially. For faster execution, you can modify the `LAYERS_TO_ANALYZE` configuration in the script to analyze only specific layers rather than all layers.

## Configuration

The script includes several configurable parameters:

- **LAYERS_TO_ANALYZE**: Specify which layers to analyze (can be model-specific or use all layers)
- **MAX_TEXTS_PER_LAYER**: Number of texts to process per layer
- **MAX_SEQUENCE_LENGTH**: Maximum sequence length for tokenization
- **Activation Thresholds**: Model-specific thresholds for detecting active neurons

## Output

The analysis generates several types of outputs:

- **Plots**: Saved in the `plots/` directory
  - N-gram head specialization across layers
  - Receptive field sensitivity plots
  - Linear decode accuracy plots
  - Gradient attribution importance plots
  - N-gram emergence patterns
  - Validation and comparison analyses

- **Logs**: Saved in the `logs/` directory
  - Raw head distributions (JSON format)
  - Complete analysis results (JSON format)
  - Kernel interpolation results (for Mamba models)

## File Structure

- **ngram_analysis.py**: Main analysis script
- **main.py**: Model setup and data loading utilities
- **utils.py**: General utility functions
- **neuron_characterization.py**: Functions for analyzing neuron behavior
- **delta_extraction.py**: Delta-sensitive neuron analysis
- **visualization_module.py**: Plotting and visualization functions
- **attention_neurons.py**: Mamba attention neuron computation

## Analysis Components

### Priority 1: Receptive Field Mass Analysis
Measures per-layer receptive field mass for 1-, 2-, and 3-token perturbations, showing how much each layer depends on recent tokens.

### Priority 2: Enhanced Linear Token Decode
Tests whether token identity can be linearly decoded from layer activations, distinguishing between representation-to-readout shifts and distributed signals.

### Priority 3: Threshold Sweep Analysis
Examines activation magnitude profiles across different thresholds to understand whether thresholding hides early detectors.

### Head-like Behavior Analysis
Groups neurons into head-like components using clustering based on trigger distributions, then analyzes their collective specialization patterns.

### Validation
Includes ablation-based validation to verify that identified head specializations have measurable functional impact.

## Model Support

The analysis supports:
- Mamba models (130M, 370M, 790M, 1.4B parameters)
- GPT-2 (small variant)

Models are automatically downloaded from Hugging Face on first use.

## Memory Considerations

The analysis can be memory-intensive, especially for larger models. The script includes memory optimization features:
- Automatic memory cleanup between layers
- Configurable text limits
- GPU memory monitoring
- Efficient batch processing

For very large models or limited GPU memory, consider:
- Reducing the number of texts processed
- Analyzing fewer layers
- Using CPU mode for smaller models

## Notes

- The analysis uses WikiText-2 dataset by default, but can work with any text corpus
- Results are saved with timestamps for tracking different runs
- The script includes error handling and will continue processing even if individual models or layers fail
- For Mamba models, the analysis includes special handling for state-space model architectures
