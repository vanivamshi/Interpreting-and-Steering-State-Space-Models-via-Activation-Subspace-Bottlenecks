# Delta Neuron Steering Framework

A comprehensive three-step framework for discovering, optimizing, and validating delta neuron steering in Mamba models.

## Overview

This framework implements a systematic approach to:
1. **Discover** beneficial delta neurons through ablation analysis
2. **Determine** optimal steering amplification factors
3. **Verify** steering performance on real-world tasks

The framework follows rigorous experimental methodology to understand and improve Mamba model performance through targeted neuron steering.

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Required packages (if not in requirements.txt):
pip install torch transformers datasets numpy matplotlib seaborn
```

### Model Setup

The framework supports various Mamba model variants. Default model: `state-spaces/mamba-130m-hf`

For other model variants, see the [Model Variants Setup](#model-variants-setup) section below.

### Using Local Models

The framework supports both HuggingFace models and local model directories. Local models are stored in the `models/` folder.

#### Models Folder Structure

The `models/` folder contains trained or custom model checkpoints:

```
models/
└── mamba_trained_on_pile/
    ├── config.json              # Model configuration
    ├── model.safetensors       # Model weights
    ├── generation_config.json  # Generation settings
    ├── tokenizer.json          # Tokenizer
    ├── tokenizer_config.json   # Tokenizer configuration
    └── special_tokens_map.json  # Special tokens mapping
```

**Note**: Training-only files (optimizer states, scheduler states, etc.) are not included as they're not needed for inference.

#### Loading Local Models

To use a local model, simply provide the path to the model directory:

```bash
# Step 1: Use local model
python step1_neuron_discovery.py \
    --model ./models/mamba_trained_on_pile \
    --save_path discovered_neurons.json

# Step 2: Use local model
python step2_steering_factor_determination.py \
    --model ./models/mamba_trained_on_pile \
    --num_examples 100

# Step 3: Use local model
python step3_steer_performance_verification.py \
    --model ./models/mamba_trained_on_pile \
    --num_samples 100 \
    --combined
```

You can use either:
- **HuggingFace models**: `--model state-spaces/mamba-130m-hf` (downloaded automatically)
- **Local models**: `--model ./models/mamba_trained_on_pile` (uses local directory)

The `mamba_model_loader.py` automatically detects if the path is a local directory or a HuggingFace model name.

#### Adding Your Own Local Models

To add a new local model:

1. Create a directory in `models/` (e.g., `models/my_custom_model/`)
2. Place the required files:
   - `config.json` - Model configuration
   - `model.safetensors` or `pytorch_model.bin` - Model weights
   - `tokenizer.json` - Tokenizer file
   - `tokenizer_config.json` - Tokenizer configuration
   - `special_tokens_map.json` - Special tokens (optional)
   - `generation_config.json` - Generation settings (optional)

3. Use it in the step scripts:
   ```bash
   python step1_neuron_discovery.py --model ./models/my_custom_model --save_path discovered_neurons.json
   ```

## Quick Start

### Complete Workflow (HuggingFace Model)

```bash
# Step 1: Discover neurons
python step1_neuron_discovery.py \
    --model state-spaces/mamba-130m-hf \
    --save_path discovered_neurons.json \
    --validation_size 200

# Step 2: Determine optimal steering factors
python step2_steering_factor_determination.py \
    --model state-spaces/mamba-130m-hf \
    --num_examples 100

# Step 3: Verify performance
python step3_steer_performance_verification.py \
    --model state-spaces/mamba-130m-hf \
    --num_samples 100 \
    --combined
```

### Complete Workflow (Local Model)

```bash
# Step 1: Discover neurons using local model
python step1_neuron_discovery.py \
    --model ./models/mamba_trained_on_pile \
    --save_path discovered_neurons.json \
    --validation_size 200

# Step 2: Determine optimal steering factors
python step2_steering_factor_determination.py \
    --model ./models/mamba_trained_on_pile \
    --num_examples 100

# Step 3: Verify performance
python step3_steer_performance_verification.py \
    --model ./models/mamba_trained_on_pile \
    --num_samples 100 \
    --combined
```

---

## Step 1: Neuron Discovery

**File**: `step1_neuron_discovery.py`

Discovers beneficial delta neurons by testing all neurons individually and ranking their impact.

### What it does:
- Extracts delta neurons from Layer 20 using SSM activation variance
- Tests each neuron's impact by removing it and measuring performance drop
- Ranks neurons by importance
- Saves discovered neurons to JSON format

### Usage

```bash
python step1_neuron_discovery.py \
    --model state-spaces/mamba-130m-hf \
    --save_path discovered_neurons.json \
    --validation_size 200 \
    --device cuda \
    --output_dir ablation_3_results
```

### Arguments

- `--model`: Model to evaluate (default: `state-spaces/mamba-130m-hf`)
- `--save_path`: **Required** - Path to save discovered neurons JSON file
- `--validation_size`: Number of validation examples per task (default: 200)
- `--device`: Device to run on (default: `cuda`)
- `--output_dir`: Directory to save results (default: `ablation_3_results`)

### Output

Creates `discovered_neurons.json` with:
- `beneficial_neurons`: List of neuron IDs with impact >= -2.0%
- `baseline_accuracy`: Baseline accuracy percentage
- `criterion`: Selection criterion used
- `all_neuron_impacts`: Complete ranking of all neurons with their impacts

### Example Output

```json
{
  "beneficial_neurons": [4, 38, 84, 94, ...],
  "baseline_accuracy": 47.0,
  "criterion": "impact >= -2.0%",
  "all_neuron_impacts": [
    {"neuron": 4, "impact": 2.5, "accuracy": 44.5},
    {"neuron": 38, "impact": 2.1, "accuracy": 44.9},
    ...
  ]
}
```

---

## Step 2: Steering Factor Determination

**File**: `step2_steering_factor_determination.py`

Determines optimal amplification strengths for different neuron groups by testing values from 0.1x to 100x.

### What it does:
- Loads discovered neurons from Step 1
- Groups neurons by impact (High Impact >2%, Neutral -2% to +2%)
- Tests amplification values from 0.1x to 100x for each group
- Identifies optimal amplification strengths

### Usage

```bash
python step2_steering_factor_determination.py \
    --model state-spaces/mamba-130m-hf \
    --num_examples 100 \
    --device cuda \
    --output_dir cluster9_ablation_results
```

### Arguments

- `--model`: Model to evaluate (default: `state-spaces/mamba-130m-hf`)
- `--num_examples`: Number of test examples (default: 100)
- `--device`: Device to run on (default: `cuda`)
- `--output_dir`: Directory to save results (default: `cluster9_ablation_results`)
- `--max_neurons`: Maximum number of neurons to test (None = all neurons)
- `--sample_neurons`: Randomly sample neurons instead of testing all
- `--perplexity_ablation`: Run layer-wise ablation with perplexity on The Pile
- `--pile_samples`: Number of samples from The Pile for perplexity (default: 100)

### Output

Saves results to `cluster9_ablation_results/cluster9_ablation_results_all_neurons.json` containing:
- Importance ranking of all neurons
- Amplification ablation results for High Impact and Neutral neurons
- Optimal amplification values for each group

### Example Output

```
High Impact Neurons (>2%) - 19 neurons
   Testing amplification values from 0.1x to 100x
   5.0x: 47.0% (+0.0%) ✓ Strong
   7.0x: 51.0% (+4.0%) ✓ Strong
   9.0x: 54.0% (+7.0%) ✓ Very strong

Neutral Neurons (-2% to +2%) - 707 neurons
   Testing amplification values from 0.1x to 100x
   2.0x: 30.0% (-17.0%) ~ Moderate
   5.0x: 30.0% (-17.0%) ~ Strong
   7.0x: 51.0% (+4.0%) ✓ Strong
   9.0x: 54.0% (+7.0%) ✓ Very strong
```

---

## Step 3: Performance Verification

**File**: `step3_steer_performance_verification.py`

Verifies steering performance on IFEval dataset with both accuracy and perplexity metrics.

### What it does:
- Tests steering on IFEval (Instruction Following Evaluation) dataset
- Compares delta-sensitive subspace vs random vs high-variance neurons
- Tests steering at different layers (18-22)
- Measures both accuracy and perplexity

### Usage

```bash
# Combined steering analysis (recommended)
python step3_steer_performance_verification.py \
    --model state-spaces/mamba-130m-hf \
    --num_samples 100 \
    --combined

# Cluster ablation
python step3_steer_performance_verification.py \
    --model state-spaces/mamba-130m-hf \
    --num_samples 100 \
    --cluster_ablation

# Layer ablation
python step3_steer_performance_verification.py \
    --model state-spaces/mamba-130m-hf \
    --num_samples 100 \
    --layer_ablation

# Both cluster and layer ablation
python step3_steer_performance_verification.py \
    --model state-spaces/mamba-130m-hf \
    --num_samples 100 \
    --both
```

### Arguments

- `--model`: Model to evaluate (default: `state-spaces/mamba-130m-hf`)
- `--num_samples`: Number of IFEval samples to use (default: 100)
- `--device`: Device to run on (default: `cuda`)
- `--output_dir`: Directory to save results (default: `cluster_ablation_ifeval_results`)
- `--combined`: Run combined steering analysis (recommended)
- `--cluster_ablation`: Run cluster ablation
- `--layer_ablation`: Run layer-wise ablation
- `--both`: Run both cluster and layer ablation

### Output

Saves results to JSON files:
- `combined_steering_ifeval_results.json`: Combined analysis results
- `cluster_ablation_ifeval_results.json`: Cluster ablation results
- `layer_ablation_ifeval_results.json`: Layer ablation results

### Example Output

```
COMBINED STEERING ANALYSIS SUMMARY
================================================================================
Baseline: Accuracy=94.0% (no steering)

Experiment                          Configuration                    Accuracy (%)   Δ from Baseline
--------------------------------------------------------------------------------------------------------
Baseline                            No steering                      94.0%         -
Delta-Sensitive @ Layer 20          Delta-sensitive neurons, Layer 20 94.0%         +0.0% ✓ (best)
Random @ Layer 20                   Random neurons, Layer 20         94.0%         +0.0%
High-Variance @ Layer 20             High-variance neurons, Layer 20  73.0%         -21.0%
```

---

## Complete Workflow Example

```bash
# 1. Discover neurons (takes time - tests all neurons)
python step1_neuron_discovery.py \
    --model state-spaces/mamba-130m-hf \
    --save_path ablation_3_results/discovered_neurons.json \
    --validation_size 200

# 2. Determine optimal steering factors
python step2_steering_factor_determination.py \
    --model state-spaces/mamba-130m-hf \
    --num_examples 100

# 3. Verify performance on IFEval
python step3_steer_performance_verification.py \
    --model state-spaces/mamba-130m-hf \
    --num_samples 100 \
    --combined
```
---

### Missing Dependencies

```bash
pip install torch transformers datasets numpy matplotlib seaborn
```

### Dataset Loading Issues

**The Pile dataset**:
- Script automatically tries `monology/pile-uncopyrighted` first
- Falls back to `EleutherAI/pile` if needed
- Requires `trust_remote_code=True` (handled automatically)

**IFEval dataset**:
- Automatically downloaded from HuggingFace on first use
- Requires internet connection

### Slow Performance

- Step 1 can be slow (tests all neurons individually)
- Use `--max_neurons` in Step 2 to limit testing
- Reduce validation/test sizes for faster iteration
- Use subset of tasks during development

### No Significant Results

- Increase sample sizes (`--validation_size`, `--num_examples`, `--num_samples`)
- Check that model is properly loaded
- Verify discovered neurons file exists from Step 1
- Ensure model is trained/pretrained correctly

---

## Output Structure

```
ablation_4/
├── step1_neuron_discovery.py
├── step2_steering_factor_determination.py
├── step3_steer_performance_verification.py
├── mamba_model_loader.py
├── pile_dataset_loader.py
├── ablation_3_results/
│   └── discovered_neurons.json          # Step 1 output
├── cluster9_ablation_results/
│   └── cluster9_ablation_results_all_neurons.json  # Step 2 output
└── cluster_ablation_ifeval_results/
    ├── combined_steering_ifeval_results.json        # Step 3 output
    ├── cluster_ablation_ifeval_results.json
    └── layer_ablation_ifeval_results.json
```
