# Mamba Mechanistic Interpretability Framework

A comprehensive experimental framework for opening Mamba's black box through systematic mechanistic interpretability analysis.

## Overview

This framework implements a step-by-step experimental recipe for understanding the internal mechanisms of Mamba models, following rigorous scientific methodology. The approach combines multiple interpretability techniques to discover, validate, and understand the circuits that implement specific functions in Mamba models.

## Key Features

- Systematic Analysis: Step-by-step experimental methodology
- Sparse Autoencoders (SAE): Discover interpretable latent features
- Activation Patching: Test necessity and sufficiency of circuits
- Temporal Causality: Analyze long-range dependencies with Jacobian maps
- Comprehensive Visualization: Rich visualizations and reporting
- Reproducible: Deterministic seeding and detailed logging
- Scalable: From small models (0.5M) to large models (1B+)

## Installation

Clone the repository and install dependencies using the requirements.txt file.

## Quick Start

### Basic Usage

Run the main evaluation script with the trained model path.

### Training on The Pile Dataset

First, install the zstandard dependency. Then train the Mamba-130M model on The Pile dataset with configurable parameters including number of samples, epochs, batch size, and learning rate.

### Evaluation Workflow

#### Step 1: Create Classified Dataset File

Create a file with classified questions from datasets including SQuAD, HotpotQA, TriviaQA, MuSiQue, DROP, and Natural Questions. Specify the number of samples per dataset and target questions per category.

#### Step 2: Run Evaluation

You can run evaluation in three modes:
- Use only custom prompts
- Use only dataset prompts
- Use both custom and dataset prompts together

## Dataset Classification

Questions are classified into these categories:

- Simple Recall: Direct fact retrieval
- Two-Hop Reasoning: Requires 2 steps of reasoning
- Three-Hop Reasoning: Requires 3+ steps of reasoning
- Long Context (5-7 facts): Multiple facts in context
- Combined Reasoning + Memory: Requires both reasoning and memory
- Stress Test (10+ facts): Very long context with many facts
- Query Dataset Tasks: TriviaQA questions

## Dataset Mapping

- SQuAD: Simple Recall, Long Context (5-7 facts), Stress Test (10+ facts)
- Natural Questions: Long Context, Stress Test (10+ facts)
- HotpotQA: Two-Hop Reasoning, Three-Hop Reasoning
- TriviaQA: Query Dataset Tasks
- MuSiQue: Two-Hop Reasoning, Three-Hop Reasoning
- DROP: Combined Reasoning + Memory

## Prompt Generation

The system uses 100 prompts per level, optimized for Mamba models. Prompts are designed to be simple and direct questions with sequential information presentation, clear patterns that Mamba can learn, short contexts to avoid state-space model challenges, and direct recall rather than complex reasoning.

The prompt generator creates prompts across 6 difficulty levels, with each level containing 100 prompts total.

## Interpretability Analysis

The interpretability analysis has been separated into a standalone script to allow running interpretability once and reusing results, faster iteration when testing steering strategies, and sharing interpretability results across different experiments.

Run the interpretability analysis script to save results to experiment_logs/interpretability_results.json which can be reused across multiple experiments.

## Framework Components

### Main Script

The main script orchestrates the complete evaluation pipeline, supports custom prompts and dataset prompts, integrates steering methods, and generates comprehensive reports.

### Prompt Generator

Generates 100 prompts per difficulty level, optimized for Mamba models, and supports both original and generated prompts.

### Model Loader

Loads Mamba models with proper architecture initialization, handles tokenizer setup, and provides memory-efficient loading.

### Dataset Loaders

- Query Dataset Loader: Loads query datasets including SQuAD, Natural Questions, and TriviaQA
- Pile Dataset Loader: Loads The Pile dataset
- Dataset Classifier: Classifies dataset questions into categories

### Training Script

Trains Mamba models on The Pile dataset with configurable training parameters and saves trained models for evaluation.

### Steering Scripts

Multiple steering scripts are available for different Mamba variants:
- DenseMamba steering
- Hyena steering
- Mamba-2 steering
- Mamba MOE steering
- MiniPLM steering
- Original Mamba steering

## Output Structure

The framework generates comprehensive outputs in the experiment_logs directory including classified dataset questions, interpretability analysis results, capability assessment results, and other experiment results.

## Best Practices

### Start Small

Begin with small models (0.5-2M parameters), use synthetic toy tasks for controlled experiments, and validate methodology before scaling up.

### Reproducibility

Always use deterministic seeding, save all configurations and random seeds, and document hyperparameters and model versions.

### Statistical Rigor

Run control tests with random subspaces, use multiple random seeds for validation, and report effect sizes and statistical significance.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or sequence length, use CPU for smaller experiments, or process data in smaller chunks.

### Missing Dependencies

Install required packages including zstandard for dataset loading.

### Dataset Loading Issues

Datasets are downloaded from HuggingFace on first use. Check internet connection and HuggingFace access. Some datasets may take time to download.

### Model Not Found

Ensure training completed successfully and check that model path matches output directory.

## Notes

The classified dataset file is saved once and can be reused across multiple runs. Questions are classified automatically based on number of supporting facts, context length, question complexity, and explicit hop counts. TriviaQA questions are always assigned to "Query Dataset Tasks" category. DROP questions are assigned to "Combined Reasoning + Memory" category. Original prompts are preserved and marked appropriately. All prompts follow the same format for consistency.

## License

This project is licensed under the MIT License.

## Acknowledgments

Built on top of the transformers library, inspired by mechanistic interpretability research, and thanks to the Mamba team for the original model.
