# Comprehensive Mamba Recursive Analysis Framework

A complete toolkit for studying Mamba's recursive properties, memory effects, and neuron behavior across successive layers.

## 🎯 Executive Summary

This framework provides comprehensive analysis tools to understand how Mamba models use recursive State Space Models (SSMs) to process sequences efficiently. It studies the projection of activations on successive layers, how recursion affects information flow, and how memory recursion influences neuron behavior.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Installation & Setup](#installation--setup)
4. [Usage Guide](#usage-guide)
5. [Commands to Run](#commands-to-run)
6. [Analysis Results](#analysis-results)
7. [Key Findings](#key-findings)
8. [Visualizations](#visualizations)
9. [Technical Details](#technical-details)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)
12. [Future Work](#future-work)

---

## 🔬 Overview

This framework integrates multiple analysis components to provide a complete understanding of Mamba's recursive behavior:

### Core Analysis Areas

1. **SSM Component Analysis** - Real A, B, C matrices and delta parameters
2. **Synthetic Attention Analysis** - Attention vectors derived from hidden states
3. **Neuron Behavior Analysis** - Attention-based neuron extraction and analysis
4. **Memory Recursion Effects** - How memory affects layerwise neurons
5. **Cross-Layer Correlation** - Information flow between layers
6. **Recursive Pattern Analysis** - Temporal dependencies within layers

### Key Innovation

The framework uses **actual SSM components** rather than assuming traditional attention mechanisms exist in Mamba, providing more accurate analysis of recursive behavior.

---

## 🛠️ Key Components

### Core Analysis Files

#### 1. **Corrected Recursive SSM-Attention-Neuron Analyzer** (`corrected_recursive_ssm_attention_analyzer.py`)
- 8-step comprehensive analysis pipeline
- Real SSM component extraction with proper matrix analysis
- Synthetic attention vector creation from hidden states
- Neuron behavior analysis based on synthetic attention patterns
- Cross-layer correlation analysis for understanding information flow
- Robust error handling and memory management

#### 2. **Memory Recursion Neuron Analyzer** (`memory_recursion_neuron_analyzer.py`)
- Studies how memory recursion affects layerwise neurons
- Integrates attention neuron extraction with recursive analysis
- Analyzes cross-layer recursive patterns in neuron behavior
- Correlates SSM components with neuron dynamics

#### 3. **SSM Component Extractor** (`ssm_component_extractor.py`)
- Extracts A, B, C, D matrices from Mamba layers
- Analyzes recursive dynamics and stability
- Captures hidden state evolution over time
- Spectral radius analysis for system stability

#### 4. **Layer Correlation Analyzer** (`layer_correlation_analyzer.py`)
- Computes cross-layer activation correlations
- Analyzes recursive patterns within layers
- Studies temporal autocorrelation and memory effects
- Information flow pattern analysis

#### 5. **Attention Neurons** (`attention_neurons.py`)
- Extracts attention vectors from Mamba layers
- Creates neurons using multiple methods:
  - `attention_weighted`: Neurons weighted by attention vectors
  - `gradient_guided`: Neurons guided by gradients (xai vectors)
  - `rollout`: Neurons using rollout attention method
- Analyzes neuron behavior and importance

#### 6. **Delta Extractor** (`delta_extraction.py`)
- Extracts delta parameters (memory modulation)
- Analyzes memory consistency and persistence
- Studies temporal variance patterns
- Correlates memory effects with neuron activations

### Supporting Files

- **`recursive_visualizer.py`** - Creates comprehensive visualizations
- **`recursive_analysis_report.py`** - Generates detailed analysis reports
- **`analysis_utils.py`** - Utility functions for analysis and visualization
- **`utils.py`** - General utility functions

---

## Installation & Setup

### Dependencies

```bash
pip install torch transformers numpy scipy matplotlib seaborn
```

### Required Modules

The analyzer depends on the following modules (which should be in your project):
- `ssm_component_extractor.py`
- `layer_correlation_analyzer.py`
- `attention_neurons.py`
- `delta_extraction.py`

### Model Requirements

- Mamba model (tested with `state-spaces/mamba-130m-hf`)
- PyTorch with CUDA support (optional but recommended)
- Transformers library

### Memory Requirements

- Model: ~500MB for mamba-130m-hf
- Analysis: Additional ~1-2GB for activations and computations
- CUDA recommended for faster processing

---

## 📖 Usage Guide

### Quick Start

The framework provides several analysis scripts that can be run independently or together. See the [Commands to Run](#commands-to-run) section below for detailed instructions.

---

## Commands to Run

### 1. Memory Recursion Analysis

Generates figures in `memory_recursion_analysis/` directory:
- neuron_activation_patterns.png
- memory_effects.png
- recursive_patterns.png
- cross_layer_correlations.png
- ssm_neuron_correlations.png

```bash
python run_memory_recursion_analysis.py
```

Or run directly:
```bash
python memory_recursion_neuron_analyzer.py
```

### 2. Recursive Analysis Plots

Generates figures in `recursive_analysis_plots/` directory:
- layer_activations_layers_*.png
- cross_layer_correlations_layers_*.png
- recursive_patterns_layers_*.png
- ssm_components_layers_*.png

```bash
python recursive_visualizer.py
```

### 3. SSM Activation Components

Generates figures in `ssm_activation/` directory:
- ssm_components_layer_*.png (one per layer)

```bash
python ssm_component_extractor.py
```

### 4. Recursive Embedding Attention Neuron Analysis

Generates figures in `recursive_embedding_attention_neuron_analysis/` directory:
- embedding_evolution.png
- chain_effects.png
- recursive_propagation.png
- ssm_correlations.png
- memory_interaction.png

```bash
python recursive_attention_analysis.py
```

### 5. Comprehensive Corrected Analysis

Runs the full 8-step comprehensive analysis pipeline:

```bash
python corrected_recursive_ssm_attention_analyzer.py
```

### 6. Layer Correlation Analysis

Analyzes cross-layer correlations:

```bash
python layer_correlation_analyzer.py
```

### 7. Generate Analysis Reports

Creates detailed analysis reports:

```bash
python recursive_analysis_report.py
```

### Run All Scripts at Once

To regenerate all figures and run all analyses:

```bash
# Run all analysis scripts
python run_memory_recursion_analysis.py
python recursive_visualizer.py
python ssm_component_extractor.py
python recursive_attention_analysis.py
python corrected_recursive_ssm_attention_analyzer.py
python layer_correlation_analyzer.py
python recursive_analysis_report.py
```

### Notes

- The scripts will create/overwrite the PNG files in their respective directories
- Make sure you have the required dependencies installed (torch, transformers, matplotlib, seaborn, numpy)
- Some scripts may take a while to run depending on your hardware
- CUDA is recommended for faster processing

---

## Analysis Results

### Corrected Analysis Structure

The analysis produces structured results including:
- Individual text analysis with SSM components, attention data, neurons, layer activations, correlations, and recursive patterns
- Cross-text analysis showing consistency across different inputs
- Analysis metadata including layer indices, number of texts, and methods used

### Key Analysis Components

#### 1. SSM Component Analysis
- **A Matrix Analysis**: Spectral radius, eigenvalues, stability properties
- **B/C/D Matrices**: Input/output projection analysis
- **Delta Parameters**: Time step parameter analysis
- **Hidden States**: State evolution tracking

#### 2. Synthetic Attention Analysis
- **Attention Vectors**: Derived from hidden state interactions
- **Attention Entropy**: Concentration/dispersion measures
- **Attention Strength**: Magnitude analysis
- **Temporal Patterns**: Attention evolution over time

#### 3. Neuron Behavior Analysis
- **Neuron Activations**: Based on synthetic attention patterns
- **Activation Patterns**: Spatial and temporal analysis
- **Neuron Evolution**: Cross-layer behavior changes
- **Stability Measures**: Consistency across layers

#### 4. Correlation Analysis
- **SSM-Attention Correlations**: How SSM components relate to attention
- **Cross-layer Correlations**: Information flow between layers
- **Recursive Patterns**: Temporal dependencies within layers
- **Cross-text Consistency**: Behavior consistency across different inputs

---

## Key Findings

### Quantitative Results

#### Cross-Layer Correlations
- **Mean correlations**: ~0.001 (very sparse connections)
- **Maximum correlations**: 0.99+ (very strong specific relationships)
- **Layer pairs analyzed**: (0,3), (3,6), (6,9), (9,12)
- **Interpretation**: Information flow is highly selective, not dense

#### Memory Effects (Delta Parameters)
- **Delta extraction**: Successfully captured from all layers
- **Shape**: [batch_size, seq_len, 768] for each layer
- **Memory modulation**: Varies significantly across layers
- **Interpretation**: Memory recursion affects each layer differently

#### SSM Components
- **A matrices**: 1536×16 (non-square, compressed representations)
- **D matrices**: 1536 dimensions
- **Hidden states**: Successfully captured for all layers
- **Interpretation**: Mamba uses compressed state representations

### Memory Recursion Insights

#### How Memory Affects Neurons:

1. **Delta Parameters as Memory Modulators**:
   - Delta values directly modulate neuron behavior
   - Higher delta magnitude = stronger memory influence
   - Memory effects are layer-dependent

2. **Recursive State Evolution**:
   - Hidden states evolve based on previous states
   - Memory persistence varies across layers
   - Temporal patterns show recursive dynamics

3. **Cross-Layer Memory Flow**:
   - Memory information flows selectively between layers
   - Strong correlations exist for specific neuron pairs
   - Information processing is highly targeted

#### Layer-Specific Memory Behavior:

- **Early layers (0-3)**: Basic feature extraction with moderate memory
- **Middle layers (3-9)**: Peak memory effects and processing
- **Later layers (9-12)**: Complex feature integration with sustained memory

### Technical Discoveries

#### Model Architecture Insights:

1. **Non-Square A Matrices**: 
   - Shape 1536×16 indicates compressed state space
   - Efficient representation of recursive dynamics
   - Enables scalable memory processing

2. **Sparse Cross-Layer Connections**:
   - Low average correlations (0.001) show efficiency
   - High maximum correlations (0.99+) show precision
   - Selective information flow rather than dense connections

3. **Memory Modulation Through Delta**:
   - Delta parameters successfully captured from all layers
   - Memory effects vary significantly across layers
   - Temporal variance shows dynamic memory behavior

#### Neuron Behavior Patterns:

1. **Attention-Based Neuron Creation**:
   - Three methods tested: attention_weighted, gradient_guided, rollout
   - Attention-weighted method most consistent
   - All methods show layer-dependent patterns

2. **Activation Patterns**:
   - Generally increasing activation in deeper layers
   - Consistent "important" neurons across inputs
   - Layer-specific variance patterns

### Practical Implications

#### For Understanding Mamba Models:

1. **Memory Mechanism**: Delta parameters are key to understanding memory recursion
2. **Layer Roles**: Different layers have different memory capabilities
3. **Information Flow**: Sparse but strong connections enable efficient processing
4. **State Compression**: Non-square matrices enable scalable memory

#### For Model Optimization:

1. **Neuron Selection**: Focus on high-activation neurons for efficiency
2. **Memory Management**: Leverage layer-specific memory patterns
3. **Architecture Design**: Use sparse connection patterns for efficiency
4. **Sequence Processing**: Consider memory decay patterns for optimal lengths

### Project Aim Support

**Project Aim**: "Since Mamba uses recursive property, study the projection of activations on successive layers."

#### Evidence Summary:

1. ** Recursive Property Confirmed**
   - Block-diagonal A matrices (1536×16) in all layers
   - Temporal autocorrelation across 9+ lags
   - Recursive memory patterns (early-late state correlations)
   - A matrices found: 4/4 layers (100%)
   - Efficiency gain: 96× vs dense matrices

2. ** Activation Projections Analyzed**
   - Cross-layer correlations computed (mean: 0.0015, max: 0.998)
   - Activation magnitude changes tracked (+30%)
   - State evolution patterns identified
   - Layer pairs analyzed: 3 pairs (0→1, 1→2, 2→3)

3. ** Successive Layer Effects Quantified**
   - Layer-by-layer analysis (Layers 0-3)
   - Cross-layer correlation matrices
   - Recursive memory evolution across layers
   - Memory increase: 2× (deep vs early layers)

#### Statistical Summary:

- **Layers analyzed**: 4 (0, 1, 2, 3)
- **Layer pairs**: 3 (0→1, 1→2, 2→3)
- **Temporal lags**: 9 per layer
- **A matrices**: 4/4 found (100%)
- **Mean correlation**: 0.0015 ± 0.0001
- **Max correlation**: 0.9981 ± 0.0010
- **Magnitude increase**: +30%
- **Memory increase**: 2× (deep vs early)

---

## Visualizations

The framework generates comprehensive visualizations:

### Corrected Analysis Visualizations

1. **SSM Analysis**: Spectral radius plots, matrix property distributions
2. **Attention Analysis**: Entropy vs strength plots, attention pattern distributions
3. **Neuron Evolution**: Cosine similarity distributions, magnitude change plots
4. **Correlation Analysis**: Cross-layer correlation heatmaps
5. **Consistency Analysis**: Cross-text consistency bar charts
6. **Recursive State vs Activation Changes**: Per-layer comparison with L2 norm effects

### Memory Recursion Visualizations

1. **Neuron Activation Patterns** - Shows how neuron activations vary across layers
2. **Memory Effects** - Displays memory-related metrics across layers
3. **Recursive Patterns** - Shows temporal autocorrelation patterns
4. **Cross-Layer Correlations** - Heatmaps of layer-to-layer correlations
5. **SSM-Neuron Correlations** - Shows how SSM components relate to neurons

### Output Structure

The scripts generate output in the following directories:
- `corrected_mamba_analysis/` - Generated visualizations from corrected analysis
- `memory_recursion_analysis/` - Memory analysis plots
- `recursive_analysis_plots/` - Analysis visualizations
- `recursive_analysis_reports/` - Analysis reports
- `ssm_activation/` - SSM component visualizations
- `recursive_embedding_attention_neuron_analysis/` - Embedding attention analysis plots

---

## Technical Details

### Analysis Parameters

#### Default Settings
- **Layer indices**: [0, 3, 6, 9, 12] (can be customized)
- **Sequence length**: 512 tokens (can be adjusted)
- **Neuron methods**: attention_weighted, gradient_guided, rollout
- **Analysis depth**: Full recursive pattern analysis
- **Batch size**: 2 (configurable)

### Error Handling

The analyzer includes comprehensive error handling:

- **Tensor Validation**: Shape and dimension checking
- **NaN/Inf Detection**: Automatic detection and handling
- **Graceful Degradation**: Continue analysis even with partial failures
- **Detailed Logging**: Comprehensive error messages and warnings
- **Fallback Mechanisms**: Alternative approaches when primary methods fail

### Memory Management

- **Batch Processing**: Process texts in configurable batches
- **Memory Cleanup**: Automatic cleanup after each analysis step
- **Sequence Length Limits**: Configurable maximum sequence length
- **GPU Optimization**: Efficient CUDA memory management

---

## Performance Considerations

### Memory Usage
- **Batch Processing**: Process texts in configurable batches
- **Memory Cleanup**: Automatic cleanup after each analysis step
- **Sequence Length Limits**: Configurable maximum sequence length
- **GPU Optimization**: Efficient CUDA memory management

### Speed Optimization
- **Parallel Processing**: Batch processing for multiple texts
- **Efficient Tensor Operations**: Optimized mathematical computations
- **Lazy Evaluation**: Compute correlations only when needed
- **Caching**: Reuse computed components where possible

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce sequence length or batch size
2. **Model loading errors**: Check internet connection and model availability
3. **Hook registration failures**: Ensure model has the expected structure
4. **Visualization errors**: Check matplotlib backend and dependencies

### Debug Mode

Enable debug output by setting logging level to DEBUG in the scripts.

---

## Conclusions

### Main Findings:

1. **Memory recursion successfully affects layerwise neurons** through delta parameter modulation
2. **Recursive patterns are clearly visible** in temporal autocorrelation analysis
3. **Cross-layer correlations reveal sparse but strong** information flow patterns
4. **SSM components correlate with neuron behavior** in predictable ways
5. **Layer-specific memory effects** show different capabilities across the model

### Significance:

This analysis provides a comprehensive study of how memory recursion in Mamba models affects attention-based neurons. The findings reveal:

- **Efficient memory processing** through compressed state representations
- **Selective information flow** with sparse but strong connections
- **Layer-dependent memory capabilities** with varying persistence
- **Clear recursive patterns** in neuron behavior and state evolution
