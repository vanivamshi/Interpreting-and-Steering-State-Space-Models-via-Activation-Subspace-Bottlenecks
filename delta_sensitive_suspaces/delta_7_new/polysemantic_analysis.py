import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from feature_visualization import get_top_activations

def get_text_samples():
    return [
        "The stock market crashed due to economic instability.",
        "Artificial intelligence is transforming healthcare.",
        "The mitochondria is the powerhouse of the cell.",
        "He scored a goal from the halfway line.",
        "NASA launched a new rover to Mars.",
        "This function calculates the factorial recursively.",
        "She baked a delicious chocolate cake for the party.",
        "Quantum computers operate on qubits instead of bits.",
        "The novel explores themes of love and betrayal.",
        "Bananas are rich in potassium and good for health.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "Electric cars are becoming increasingly popular.",
        "The algorithm optimizes neural network weights.",
        "Shakespeare wrote many famous tragedies and comedies.",
        "Elephants are the largest land mammals.",
    ]


def cluster_contexts(results, neuron_index=None, layer_idx=None, num_clusters=3, averaged=False):
    """
    Cluster activation contexts to analyze polysemanticity.
    
    Args:
        results: list of (activation, token, decoded_word, decoded_context, full_text)
        neuron_index: int, neuron index for logging
        layer_idx: int, layer index for logging
        num_clusters: number of semantic clusters to form
        averaged: bool, whether this is averaged across layers
    """
    # Extract contexts for clustering - using decoded_context (4th element)
    contexts = [r[3] for r in results]  # decoded_context
    
    # Handle empty or invalid contexts
    valid_contexts = [ctx for ctx in contexts if ctx and len(ctx.strip()) > 0]
    if len(valid_contexts) < num_clusters:
        print(f"Warning: Only {len(valid_contexts)} valid contexts found, reducing clusters to {len(valid_contexts)}")
        num_clusters = max(1, len(valid_contexts))
    
    if not valid_contexts:
        print("No valid contexts found for clustering!")
        return {}

    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(valid_contexts)
        
        if X.shape[0] < num_clusters:
            num_clusters = X.shape[0]
            
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(X)
        
        # Map labels back to original results
        context_to_label = {}
        valid_idx = 0
        for i, ctx in enumerate(contexts):
            if ctx and len(ctx.strip()) > 0:
                context_to_label[ctx] = kmeans.labels_[valid_idx]
                valid_idx += 1
            else:
                context_to_label[ctx] = -1  # Invalid context
        
        labels = [context_to_label[ctx] for ctx in contexts]
        
    except Exception as e:
        print(f"Error in clustering: {e}")
        # Fallback: assign all to one cluster
        labels = [0] * len(results)
        num_clusters = 1

    clustered = {}
    for label, r in zip(labels, results):
        if label >= 0:  # Skip invalid contexts
            clustered.setdefault(label, []).append(r)

    # Build report content
    report_lines = []
    report_lines.append("\n=== Polysemantic Clustering Report ===")
    if averaged:
        if neuron_index is not None:
            report_lines.append(f"Neuron {neuron_index} (Averaged Across All Layers)")
        else:
            report_lines.append("Averaged Across All Layers")
    elif neuron_index is not None and layer_idx is not None:
        report_lines.append(f"Neuron {neuron_index} in Layer {layer_idx}")
    elif neuron_index is not None:
        report_lines.append(f"Neuron {neuron_index}")
    
    for label, group in clustered.items():
        report_lines.append(f"\n🔹 Cluster {label + 1} — {len(group)} activations")
        unique_texts = set()
        
        # Unpack the correct tuple structure: (val, token, word, context, full_text)
        for val, token, word, context, full_text in group:
            line = f"  Token: '{token}' | Word: '{word}' | Act: {val:.3f} | Context: ...{context}..."
            report_lines.append(line)
            unique_texts.add(full_text)
        
        report_lines.append("  📄 Text Sources:")
        for t in list(unique_texts)[:3]:  # Print first 3 unique texts
            report_lines.append(f"    - {t[:120]}{'...' if len(t) > 120 else ''}")
    
    # Print to terminal
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    os.makedirs("images", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if averaged:
        if neuron_index is not None:
            filename = f"images/polysemantic_analysis_neuron_{neuron_index}_averaged_{timestamp}.txt"
        else:
            filename = f"images/polysemantic_analysis_averaged_{timestamp}.txt"
    elif neuron_index is not None and layer_idx is not None:
        filename = f"images/polysemantic_analysis_neuron_{neuron_index}_layer{layer_idx}_{timestamp}.txt"
    elif neuron_index is not None:
        filename = f"images/polysemantic_analysis_neuron_{neuron_index}_{timestamp}.txt"
    else:
        filename = f"images/polysemantic_analysis_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("POLYSEMANTIC CLUSTERING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(report_text)
        f.write("\n\n")
        f.write("=" * 50 + "\n")
        f.write("End of Report\n")
    
    print(f"✅ Saved polysemantic analysis report to {filename}")
    
    return clustered

def plot_top_activations(results, neuron_index, layer_idx, averaged=False):
    """Plot top activations for a neuron."""
    # Extract tokens and values from the 5-tuple structure
    tokens = [r[1] for r in results]  # token
    values = [r[0] for r in results]  # activation value
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=tokens, orient="h")
    title_suffix = " (Averaged Across All Layers)" if averaged else f" (Layer {layer_idx})"
    plt.title(f"Top Activations for Neuron {neuron_index}" + title_suffix, fontsize=20, fontweight='bold')
    plt.xlabel("Activation Value", fontsize=18, fontweight='bold')
    plt.ylabel("Token", fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    
    # Save the plot to images folder
    import os
    os.makedirs("images", exist_ok=True)
    if averaged:
        plt.savefig(f"images/polysemantic_activations_neuron_{neuron_index}.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved averaged polysemantic activations plot to images/polysemantic_activations_neuron_{neuron_index}.png")
    else:
        plt.savefig(f"images/polysemantic_activations_neuron_{neuron_index}_layer{layer_idx}.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved polysemantic activations plot to images/polysemantic_activations_neuron_{neuron_index}_layer{layer_idx}.png")
    plt.close()
    # plt.show()  # Not needed with non-interactive backend


def setup_model(model_name="state-spaces/mamba-130m-hf"):
    """Setup model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def run_polysemantic_analysis(neuron_index=560, layer_idx=0, return_data_only=False):
    """Run complete polysemantic analysis for a specific neuron."""
    model, tokenizer = setup_model()
    texts = get_text_samples()
    
    print(f"\nExtracting top activations for Neuron {neuron_index} (Layer {layer_idx})...")
    results = get_top_activations(model, tokenizer, texts, neuron_index=neuron_index, layer_idx=layer_idx)
    
    if return_data_only:
        return results
    
    print(f"\nVisualizing top activations:")
    for i, (val, token, word, context, _) in enumerate(results):
        print(f"{i+1:2d}. Token: '{token}' | Word: '{word}' | Act: {val:.3f} | Context: ...{context}...")
    
    plot_top_activations(results, neuron_index, layer_idx)
    cluster_contexts(results, neuron_index=neuron_index, layer_idx=layer_idx, num_clusters=3, averaged=False)


if __name__ == "__main__":
    run_polysemantic_analysis(neuron_index=560, layer_idx=0)