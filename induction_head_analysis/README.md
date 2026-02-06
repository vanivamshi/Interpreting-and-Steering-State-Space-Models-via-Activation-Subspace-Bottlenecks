# Mamba Induction Head Analysis

This project analyzes induction head behavior in Mamba models by fine-tuning a Mamba model on the Wikitext-2 dataset and then performing comprehensive induction pattern analysis.

## Overview

The project investigates how Mamba models detect and utilize induction patterns (repeated sequences) in text. It includes:

- Fine-tuning Mamba-130M on Wikitext-2 dataset
- Induction head analysis with visualization
- Layer-wise induction pattern emergence analysis
- Investigation of the gap between induction detection and prediction accuracy
- Comprehensive visualization and analysis reports

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- PyTorch
- Transformers library
- Datasets library
- NumPy
- Matplotlib

## Installation

1. Clone or navigate to this directory:
```bash
cd /home/HDD/ATAF/vamshi/LLM_paper/induction_1_new
```

2. Install required dependencies:
```bash
pip install torch transformers datasets numpy matplotlib
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main analysis script:

```bash
python main_attention.py
```

Or make it executable and run directly:

```bash
chmod +x main_attention.py
./main_attention.py
```

### What the Script Does

1. **Loads Dataset**: Downloads and loads the Salesforce/wikitext-2-v1 dataset
2. **Loads Model**: Downloads the `state-spaces/mamba-130m-hf` model from HuggingFace
3. **Fine-tunes Model**: Fine-tunes the model on the dataset (1 epoch by default)
4. **Induction Analysis**: Performs comprehensive induction head analysis including:
   - Synthetic induction sequence testing
   - Real text sample analysis
   - Layer-wise induction pattern emergence
   - Prediction accuracy analysis
5. **Investigation**: Investigates why strong induction scores may not translate to accurate predictions
6. **Generates Outputs**: Creates visualizations and analysis summaries

### Output Files

The script generates the following outputs in the `plots/` directory:

- `induction_analysis_seq{idx}_{timestamp}.png` - Induction analysis visualizations for each test sequence
- `layer_wise_induction_{timestamp}.png` - Layer-wise induction pattern emergence
- `prediction_gap_seq{idx}_{timestamp}.png` - Analysis of prediction gaps
- `analysis_summary_{timestamp}.txt` - Comprehensive text summary of all analysis results

### Configuration

You can modify the following parameters in `main_attention.py`:

- **Dataset size**: Change `texts = texts[:1000]` to use more/fewer samples
- **Training epochs**: Modify `num_epochs=1` in the `fine_tune_model()` call
- **Batch size**: Adjust `batch_size=1` (reduce if running out of memory)
- **Sequence length**: Change `max_length=256` for longer/shorter sequences
- **Learning rate**: Modify `learning_rate=5e-5`
- **Gradient accumulation**: Adjust `gradient_accumulation_steps=4`

### Memory Optimization

The script includes memory optimizations:
- Small batch size (1) with gradient accumulation
- CUDA cache clearing
- Reduced sequence length (256 tokens)
- Gradient clipping

If you encounter out-of-memory errors:
- Reduce `max_length`
- Reduce `batch_size` (already at 1)
- Reduce dataset size
- Use CPU instead of GPU (slower but uses less memory)

## Project Structure

```
induction_1_new/
├── main_attention.py          # Main entry point script
├── induction.py                # MambaInductionAnalyzer class and analysis functions
├── induction_investigation.py  # Investigation of induction-prediction gap
└── README.md                   # This file
```

## Key Features

### Induction Analysis
- Detects repeated patterns in sequences
- Computes induction scores based on hidden state similarities
- Visualizes induction head activations
- Analyzes state similarity matrices

### Layer-wise Analysis
- Tracks how induction patterns emerge across model layers
- Visualizes induction scores at different layers
- Identifies which layers contribute most to induction detection

### Prediction Gap Investigation
- Analyzes why high induction scores may not lead to accurate predictions
- Compares logit distributions at high vs low induction positions
- Investigates the transformation from hidden states to predictions
- Compares first vs second occurrence contexts

## Example Output

The script will print progress information and generate:
- Console output showing training progress and analysis results
- Visualization plots saved to `plots/` directory
- Detailed text summary with all numerical results

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_length` parameter
- Reduce dataset size
- Use CPU: The script automatically falls back to CPU if CUDA is unavailable

### Dataset Download Issues
- The script includes fallback dummy data if dataset download fails
- Check internet connection for HuggingFace dataset access

### Model Download Issues
- Ensure you have internet access to download from HuggingFace
- The model `state-spaces/mamba-130m-hf` will be cached after first download
