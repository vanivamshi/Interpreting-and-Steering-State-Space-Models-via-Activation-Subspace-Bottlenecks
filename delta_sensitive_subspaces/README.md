# Delta 7 Analysis

## Overview

This project provides comprehensive neuron analysis tools for Mamba and Transformer models, with a focus on understanding knowledge representation, attention mechanisms, and model behavior.

## Features

### Attention Weights Integration

The project integrates attention weights functionality for analyzing Mamba models. This includes:

- **Attention Analysis**: Runs attention analysis for Mamba models using attention weights
- **Multiple Methods**: Supports different neuron creation methods including attention-weighted, gradient-guided, and rollout approaches
- **Visualization**: Creates comprehensive visualizations of attention neurons and their behaviors
- **Knowledge Extraction**: Analyzes how attention weights contribute to knowledge extraction

### Analysis Capabilities

The main analysis pipeline includes:

- **Delta Extraction**: Finds delta-sensitive neurons that respond to knowledge changes
- **Bottleneck Analysis**: Identifies information bottlenecks in the model
- **Feature Visualization**: Visualizes top activations for specific neurons
- **Polysemantic Analysis**: Analyzes neurons that respond to multiple contexts
- **Conflict Resolution**: Studies how models handle conflicting information
- **Bias Analysis**: Detects and analyzes bias in neuron responses
- **Delta Intervention**: Tests the effects of suppressing or amplifying knowledge neurons

### Automatic Model Detection

The system automatically detects Mamba models by checking for the mixer attribute in layers, and only runs attention analysis on compatible models.

## Usage

### Basic Usage

Run analysis on default models (Mamba and GPT-2):

```
python main.py
```

### Custom Models

Specify custom models to analyze:

```
python main.py --models state-spaces/mamba-130m-hf gpt2
```

### Attention Analysis

Enable attention weight analysis for Mamba models:

```
python main.py --models state-spaces/mamba-130m-hf --enable_attention
```

### Custom Configuration

Configure specific layers and methods for attention analysis:

```
python main.py --models state-spaces/mamba-130m-hf \
    --enable_attention \
    --attention_layers 0 3 6 9 \
    --attention_methods attention_weighted gradient_guided
```

### Analysis Parameters

- `--layer`: Layer index to analyze (default: 1)
- `--top_k`: Number of top neurons to analyze (default: 10)
- `--samples`: Number of text samples to use (default: 50)
- `--save_results`: Save analysis results to JSON file
- `--use_cached_neurons`: Path to JSON file with precomputed neuron indices

## Output

### Visualizations

The analysis generates various visualizations saved in the `images/` directory:

- Bottleneck analysis plots showing perturbation effects
- Delta intervention comparisons (suppression and amplification)
- State interpolation analysis showing conflict resolution patterns
- Bias sensitivity heatmaps
- Polysemantic activation visualizations
- Attention neuron visualizations (for Mamba models)

### Results

Analysis results can be saved as JSON files containing:

- Delta-sensitive neuron rankings
- Perturbation effects
- Comprehensive bottleneck analysis
- Information flow analysis
- Attention analysis results (for Mamba models)
- Knowledge extraction metrics
- Conflict resolution patterns
- Bias detection results

## Integration Benefits

### Enhanced Analysis

- Provides deeper insights into Mamba model behavior
- Combines traditional neuron analysis with attention weights
- Enables knowledge extraction analysis

### Consistent Interface

- Follows established analysis patterns
- Integrates seamlessly with existing analysis pipeline
- Maintains backward compatibility

### Flexible Configuration

- Configurable layers and methods via command line
- Easy to enable/disable attention analysis
- Customizable analysis parameters

### Comprehensive Output

- Rich visualizations for analysis
- Detailed results for further processing
- Integration with existing result saving

## Project Structure

The main analysis is orchestrated by `main.py`, which imports functionality from:

- `utils.py`: Model structure utilities
- `delta_extraction.py`: Delta-sensitive neuron finding
- `bottleneck_analysis.py`: Bottleneck identification
- `feature_visualization.py`: Feature visualization
- `polysemantic_analysis.py`: Polysemantic neuron analysis
- `conflicting_information.py`: Conflict resolution analysis
- `bias_analysis.py`: Bias detection
- `delta_intervention_analysis.py`: Intervention experiments
- `attention_neurons.py`: Attention weight analysis for Mamba models

## Key Features

### Multiple Analysis Methods

The attention analysis supports three methods:

- **attention_weighted**: Neurons weighted by attention vectors
- **gradient_guided**: Neurons guided by gradients (XAI vectors)
- **rollout**: Neurons using rollout attention method

### Comprehensive Analysis Pipeline

The system performs:

- Extraction of attention vectors from multiple layers
- Creation of mamba neurons using different methods
- Analysis of neuron behavior and importance
- Knowledge extraction insights
- Averaging results across layers for robust analysis

### Rich Visualizations

The visualization system creates:

- Neuron activation plots
- Attention heatmaps
- Importance score visualizations
- Top neuron comparisons
- Averaged results across layers

