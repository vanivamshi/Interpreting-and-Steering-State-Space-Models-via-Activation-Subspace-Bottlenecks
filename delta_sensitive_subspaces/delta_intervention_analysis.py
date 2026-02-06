# delta_intervention_analysis.py

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from typing import List, Dict
from delta_extraction import register_perturbation_hook
from utils import get_model_layers
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_answer_probability(model, tokenizer, prompt, answer, device):
    """
    Compute probability of `answer` appearing after `prompt`.
    """
    model.eval()
    # Ensure model is on the correct device
    model = model.to(device)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits

    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]

    start_idx = inputs["input_ids"].shape[1] - len(answer_tokens)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total_logprob = 0.0
    for i, token_id in enumerate(answer_tokens):
        token_logprob = log_probs[0, start_idx + i - 1, token_id].item()
        total_logprob += token_logprob

    return np.exp(total_logprob)


def get_probability_with_intervention(model, tokenizer, prompt, answer, neuron_indices, layer_idx, mode, std=1.0):
    """
    Compute probability after perturbing neurons.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layers = get_model_layers(model)
    target_layer = layers[layer_idx]

    hook = register_perturbation_hook(
        target_layer, neuron_indices, mode=mode, std=std
    )
    prob = get_answer_probability(model, tokenizer, prompt, answer, device)
    hook.remove()

    return prob


def run_delta_intervention_analysis(model, tokenizer, top_neurons, layer_idx, relations_dict, baseline_model_name="gpt2", return_data_only=False):
    """
    Run intervention analysis for delta neurons:
    - suppression (zeroing)
    - amplification (scaling)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure model is on the correct device
    model = model.to(device)

    # Load transformer baseline
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_name).to(device)

    suppression_changes = {}
    amplification_changes = {}

    for relation, prompts in relations_dict.items():
        baseline_probs = []
        suppressed_probs = []
        amplified_probs = []

        for prompt, answer in prompts:
            # Baseline transformer probability
            p_base = get_answer_probability(baseline_model, baseline_tokenizer, prompt, answer, device)

            # Suppression on your model
            p_suppress = get_probability_with_intervention(
                model, tokenizer, prompt, answer,
                top_neurons, layer_idx,
                mode="zero"
            )

            # Amplification on your model
            p_amplify = get_probability_with_intervention(
                model, tokenizer, prompt, answer,
                top_neurons, layer_idx,
                mode="scale",
                std=2.0
            )

            baseline_probs.append(p_base)
            suppressed_probs.append(p_suppress)
            amplified_probs.append(p_amplify)

        baseline_mean = np.mean(baseline_probs)
        suppression_mean = np.mean(suppressed_probs)
        amplification_mean = np.mean(amplified_probs)

        suppression_pct_change = ((suppression_mean - baseline_mean) / baseline_mean) * 100
        amplification_pct_change = ((amplification_mean - baseline_mean) / baseline_mean) * 100

        suppression_changes[relation] = {"baseline_mean": baseline_mean,"model_mean": suppression_mean,"pct_change": suppression_pct_change}
        amplification_changes[relation] = {"baseline_mean": baseline_mean,"model_mean": amplification_mean,"pct_change": amplification_pct_change}

    if return_data_only:
        return suppression_changes, amplification_changes

    # Plot suppression (Figure 4)
    relations = list(suppression_changes.keys())
    baseline_means = [suppression_changes[r]["baseline_mean"] for r in relations]
    model_means = [suppression_changes[r]["model_mean"] for r in relations]

    x = np.arange(len(relations))
    width = 0.35

    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, baseline_means, width, label="Transformer Baseline", color="gray")
    plt.bar(x + width/2, model_means, width, label="Mamba Suppressed", color="red")
    plt.xticks(x, relations, rotation=45, ha='right', fontsize=16)
    plt.ylabel("Answer Probability", fontsize=18, fontweight='bold')
    plt.title(f"Effect of Knowledge Neuron Suppression: Mamba vs Transformer (Layer {layer_idx})", fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    
    # Save the plot to images folder
    import os
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/delta_intervention_suppression_layer{layer_idx}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved suppression plot to images/delta_intervention_suppression_layer{layer_idx}.png")
    plt.close()
    # plt.show()  # Not needed with non-interactive backend

    # Plot amplification (Figure 5)
    relations = list(amplification_changes.keys())
    baseline_means = [amplification_changes[r]["baseline_mean"] for r in relations]
    model_means = [amplification_changes[r]["model_mean"] for r in relations]

    x = np.arange(len(relations))

    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, baseline_means, width, label="Transformer Baseline", color="gray")
    plt.bar(x + width/2, model_means, width, label="Mamba Amplified", color="green")
    plt.xticks(x, relations, rotation=45, ha='right', fontsize=16)
    plt.ylabel("Answer Probability", fontsize=18, fontweight='bold')
    plt.title(f"Effect of Knowledge Neuron Amplification: Mamba vs Transformer (Layer {layer_idx})", fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    
    # Save the plot to images folder
    plt.savefig(f"images/delta_intervention_amplification_layer{layer_idx}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved amplification plot to images/delta_intervention_amplification_layer{layer_idx}.png")
    plt.close()
    # plt.show()  # Not needed with non-interactive backend

    print("Suppression Changes (%):", suppression_changes)
    print("Amplification Changes (%):", amplification_changes)

    return suppression_changes, amplification_changes


def plot_delta_intervention_averaged(all_suppression_changes: List[Dict], all_amplification_changes: List[Dict]):
    """Plot averaged delta intervention results across all layers."""
    if not all_suppression_changes or not all_amplification_changes:
        return
    
    # Get all relations from first result
    relations = list(all_suppression_changes[0].keys())
    
    # Average across layers
    avg_suppression = {}
    avg_amplification = {}
    
    for relation in relations:
        baseline_means = [sc[relation]["baseline_mean"] for sc in all_suppression_changes if relation in sc]
        model_means_supp = [sc[relation]["model_mean"] for sc in all_suppression_changes if relation in sc]
        model_means_amp = [ac[relation]["model_mean"] for ac in all_amplification_changes if relation in ac]
        
        avg_suppression[relation] = {
            "baseline_mean": np.mean(baseline_means),
            "model_mean": np.mean(model_means_supp)
        }
        avg_amplification[relation] = {
            "baseline_mean": np.mean(baseline_means),
            "model_mean": np.mean(model_means_amp)
        }
    
    # Plot suppression
    relations = list(avg_suppression.keys())
    baseline_means = [avg_suppression[r]["baseline_mean"] for r in relations]
    model_means = [avg_suppression[r]["model_mean"] for r in relations]
    
    x = np.arange(len(relations))
    width = 0.35
    
    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, baseline_means, width, label="Transformer Baseline", color="gray")
    plt.bar(x + width/2, model_means, width, label="Mamba Suppressed", color="red")
    plt.xticks(x, relations, rotation=45, ha='right', fontsize=16)
    plt.ylabel("Answer Probability", fontsize=18, fontweight='bold')
    plt.title("Effect of Knowledge Neuron Suppression: Mamba vs Transformer (Averaged Across All Layers)", fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    
    import os
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/delta_intervention_suppression.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved averaged suppression plot to images/delta_intervention_suppression.png")
    plt.close()
    
    # Plot amplification
    baseline_means = [avg_amplification[r]["baseline_mean"] for r in relations]
    model_means = [avg_amplification[r]["model_mean"] for r in relations]
    
    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, baseline_means, width, label="Transformer Baseline", color="gray")
    plt.bar(x + width/2, model_means, width, label="Mamba Amplified", color="green")
    plt.xticks(x, relations, rotation=45, ha='right', fontsize=16)
    plt.ylabel("Answer Probability", fontsize=18, fontweight='bold')
    plt.title("Effect of Knowledge Neuron Amplification: Mamba vs Transformer (Averaged Across All Layers)", fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    
    plt.savefig(f"images/delta_intervention_amplification.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved averaged amplification plot to images/delta_intervention_amplification.png")
    plt.close()
