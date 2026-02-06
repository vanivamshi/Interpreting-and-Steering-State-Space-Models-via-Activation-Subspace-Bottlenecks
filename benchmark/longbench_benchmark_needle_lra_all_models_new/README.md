# Mamba Benchmark Framework

A comprehensive benchmarking framework for evaluating Mamba models and variants on LongBench, RULER, and LRA benchmarks.

## Overview

This framework implements systematic benchmarking of Mamba models and enhanced variants (mamba_deeptrace) across multiple evaluation tasks. The approach combines multiple benchmark suites to provide comprehensive performance analysis.

## Key Features

- **Benchmarking Suite**: LongBench, RULER, and LRA evaluation
- **Model Variants**: Support for Mamba, mamba_deeptrace, SteeredMamba, Hyena, and more
- **Performance Metrics**: Accuracy, confidence, calibration, faithfulness, robustness, latency, memory
- **Real Dataset Support**: Training on The Pile dataset with evaluation on benchmarks
- **Comprehensive Analysis**: Multi-task evaluation across different model architectures

## Main Files

- `benchmark_optimized_weights_real_dataset.py` - Main benchmarking script with real dataset support
- `benchmark_optimized_weights_working.py` - Working version of benchmark script
- `mamba2_layer.py` - Core mamba_deeptrace layer implementation
- `mamba2_context_fix.py` - Context-aware fixes
- `mamba2_safe_fix.py` - Safe scaling implementations
- `mamba2_final_solution.py` - Optimized context scaling
- `mamba2_simple_qa_fix.py` - Balanced context scaling for QA tasks
- `mamba2_ruler_fix.py` - RULER benchmark optimizations

## Installation

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

Run the main benchmark script:

```bash
python benchmark_optimized_weights_real_dataset.py
```

## Benchmark Results

### LongBench Results (Pile Trained)
- GPT2: 95.00% accuracy, 12.18% confidence
- Mamba: 100.00% accuracy, 20.75% confidence
- mamba_deeptrace: 100.00% accuracy, 32.35% confidence (best calibration: 98.92%)
- SteeredMamba: 100.00% accuracy, 36.61% confidence (best calibration: 99.73%)
- Mamba2Internet: 100.00% accuracy, 16.71% confidence

### RULER Benchmark Results
- GPT2: 94.00% NIAH, 59.00% Aggregation, 58.00% QA
- Mamba: 84.00% NIAH, 67.00% Aggregation, 75.00% QA
- mamba_deeptrace: 90.00% NIAH, 37.00% Aggregation, 28.00% QA
- Mamba2Internet: 70.43% NIAH, 40.00% Aggregation, 89.00% QA (best QA)

### LRA Benchmark Results
- mamba_deeptrace achieves 95.00% on Pathfinder task (excellent performance)
- Mamba leads with 62.00% on ListOps
- Various models show strengths across different tasks

## Key Improvements

After Pile training:
- mamba_deeptrace confidence: 32.35% (vs baseline ~20%)
- SteeredMamba confidence: 36.61%
- mamba_deeptrace calibration: 98.92%
- SteeredMamba calibration: 99.73%
- mamba_deeptrace Pathfinder: 95.00% (excellent)

## Implementation Summary

The framework includes:
- Complete benchmarking pipeline for multiple model architectures
- Support for real dataset training and evaluation
- Comprehensive performance metrics across multiple benchmarks
- Optimized implementations for different task types
- Memory-efficient processing for large models

## Activation Collection Guide

### Common Issues and Solutions

**Model Layer Access Issues**: Use direct layer access (`model.layers` or `model.backbone.layers`) instead of utility functions.

**Hook Registration Failures**: Verify layer exists before registering hooks, use proper error handling.

**Activation Shape Inconsistencies**: Validate activation shapes before processing, handle variable sequence lengths properly.

**Memory Issues**: Process texts in smaller batches, use gradient checkpointing, clear activations between runs.

### Validation Checklist

- Model loads successfully
- Tokenizer works with test texts
- Model layers are accessible
- Hooks register successfully
- Forward pass completes without errors
- Activations are captured with expected properties
- No NaN or Inf values

## Activation Extraction Updates

All analysis modules use direct model layer access for better compatibility:
- Primary strategy: `model.layers`
- Fallback strategy: `model.backbone.layers`
- Improved error handling with clear logging
- Graceful degradation when extraction fails

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size, use CPU for smaller experiments, process data in smaller chunks.

**Model Loading Errors**: Check CUDA state, reset CUDA cache, verify model compatibility.

**Activation Collection Issues**: Verify model structure, test with simple inputs first, check hook registration.

## Requirements

See `requirements.txt` for full list of dependencies including:
- torch >= 2.0.0
- transformers >= 4.30.0
- numpy, pandas, matplotlib
- datasets, bert_score
- And other required packages

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built on top of the transformers library
- Thanks to the Mamba team for the original model
- Benchmark datasets: LongBench, RULER, LRA
