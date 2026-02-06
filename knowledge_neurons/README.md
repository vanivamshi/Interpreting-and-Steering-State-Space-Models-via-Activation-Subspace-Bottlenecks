# Knowledge Neurons Analysis

This project performs knowledge neuron analysis using integrated gradients on GPT-2 and Mamba language models. It identifies neurons responsible for storing factual knowledge and analyzes their overlap patterns across different facts and relations.

## Features

- **Knowledge Neuron Detection**: Uses integrated gradients to identify neurons that encode factual knowledge
- **Multi-Model Support**: Analyzes both GPT-2 and Mamba models
- **Attention Analysis**: For Mamba models, includes attention weight mechanism integration
- **Overlap Analysis**: Computes Jaccard similarity and intersection statistics for knowledge neurons across facts
- **Layer-wise Analysis**: Analyzes knowledge neurons across all model layers

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Transformers (Hugging Face)
- CUDA (optional, for GPU acceleration)

## Installation

1. Install the required dependencies:

```bash
pip install torch numpy transformers
```

2. Ensure you have sufficient disk space and memory to download and run the models (GPT-2 and Mamba-130M).

## Usage

The main script `main.py` supports three analysis modes:

### 1. Comprehensive Analysis (Default)

Runs full analysis including knowledge neurons and attention mechanisms for Mamba models:

```bash
python main.py comprehensive
```

or simply:

```bash
python main.py
```

This will:
- Analyze the Mamba-130M model
- Run knowledge neuron detection
- Perform attention analysis (for Mamba models)
- Save results to `mamba_comprehensive_analysis.json`

### 2. Attention Analysis

Runs knowledge neurons + attention analysis for Mamba models:

```bash
python main.py attention
```

This will:
- Analyze the Mamba-130M model
- Run both knowledge neuron and attention analysis
- Save results to `mamba_analysis.json`

### 3. Knowledge Neurons Only

Runs knowledge neuron analysis for both GPT-2 and Mamba models:

```bash
python main.py knowledge
```

This will:
- Analyze both GPT-2 and Mamba-130M models
- Compare knowledge neuron overlap statistics
- Save results to:
  - `gpt2_knowledge_neurons.json`
  - `mamba_knowledge_neurons.json`

## Output

The analysis generates JSON files containing:

- **Knowledge neurons**: Attribution scores for each neuron per fact
- **Layer-wise results**: Knowledge neurons identified at each layer
- **Overlap statistics**: Jaccard similarity and intersection sizes for:
  - Intra-relation pairs (facts from the same relation type)
  - Inter-relation pairs (facts from different relation types)
- **Attention analysis** (for Mamba models): Attention-weighted neuron activations

## Facts Analyzed

The analysis tests knowledge neurons on various factual relations:

- **Geography**: Capital cities (France, Germany, Italy), landmarks (Eiffel Tower), rivers (Nile, Amazon, Yangtze)
- **History**: World War II end year
- **Chemistry**: Water chemical formula
- **Literature**: Authors of plays (Romeo and Juliet, Hamlet, Macbeth)

## Project Structure

```
.
├── main.py                 # Main entry point
├── knowledge_neurons.py    # Knowledge neuron detection using integrated gradients
├── attention_neurons.py    # Attention-based neuron analysis for Mamba models
└── README.md              # This file
```

## Example Output

The script prints detailed statistics including:

- Number of significant neurons per fact
- Average Jaccard similarity for intra-relation vs inter-relation pairs
- Layer-wise analysis results
- Attention analysis summary (for Mamba models)

## Notes

- The analysis may take some time depending on your hardware
- Models are automatically downloaded from Hugging Face on first run
- GPU acceleration is recommended for faster execution
- Results are saved as JSON files for further analysis

## Troubleshooting

- **Out of memory errors**: Try running on a smaller model or reduce the number of facts analyzed
- **Model download issues**: Ensure you have internet connectivity and sufficient disk space
- **CUDA errors**: The script will automatically fall back to CPU if CUDA is not available
