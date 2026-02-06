import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class MambaAttentionNeurons:
    """
    Creates mamba neurons based on attention vectors from Mamba models.
    This implementation is based on the HiddenMambaAttn approach which
    views Mamba models as attention-driven models.
    """
    def __init__(self, model, enable_attention_computation=True):
        self.model = model
        self.enable_attention_computation = enable_attention_computation
        self.attention_matrices = {}
        self.xai_vectors = {}

        # Enable attention matrix computation if supported
        # Handle different model structures
        layers = None
        if hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            layers = self.model.backbone.layers
        
        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'compute_attn_matrix'):
                    layer.mixer.compute_attn_matrix = enable_attention_computation
    
    def extract_attention_vectors(self, inputs, layer_indices: Optional[List[int]] = None):
        """
        Extract attention vectors from specified layers of the Mamba model.
        
        Args:
            inputs: Input tensor to the model
            layer_indices: List of layer indices to extract from. If None, extracts from all layers.
        
        Returns:
            Dictionary containing attention vectors for each layer
        """
        # Get the correct layers reference
        layers = None
        if hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            layers = self.model.backbone.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        
        if layer_indices is None:
            if layers is not None:
                layer_indices = list(range(len(layers)))
            else:
                print("Warning: Could not determine number of layers, using default [0]")
                layer_indices = [0]
        
        # Forward pass to compute attention matrices
        with torch.no_grad():
            outputs = self.model(inputs)
        
        attention_data = {}
        
        if layers is None:
            print("Warning: Could not find model layers")
            return attention_data
        
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                if hasattr(layer, 'mixer'):
                    mixer = layer.mixer
                    
                    # Extract attention matrices if computed
                    if hasattr(mixer, 'attn_matrix_a') and hasattr(mixer, 'attn_matrix_b'):
                        attn_a = mixer.attn_matrix_a.detach()
                        attn_b = mixer.attn_matrix_b.detach()
                        
                        # Combine attention matrices (similar to HiddenMambaAttn approach)
                        combined_attention = (attn_a + attn_b) / 2.0
                        
                        # Extract attention vectors (average across heads)
                        attention_vectors = combined_attention.mean(dim=1)  # Average across attention heads
                        
                        attention_data[layer_idx] = {
                            'attention_matrix': combined_attention,
                            'attention_vectors': attention_vectors,
                            'attn_matrix_a': attn_a,
                            'attn_matrix_b': attn_b
                        }
                    
                    # Extract xai vectors if available
                    if hasattr(mixer, 'xai_b'):
                        xai_vectors = mixer.xai_b.detach()
                        if layer_idx not in attention_data:
                            attention_data[layer_idx] = {}
                        attention_data[layer_idx]['xai_vectors'] = xai_vectors
        
        return attention_data
    
    def create_mamba_neurons(self, attention_data: dict, method: str = 'attention_weighted'):
        """
        Create mamba neurons based on attention vectors.
        
        Args:
            attention_data: Dictionary containing attention data from extract_attention_vectors
            method: Method to create neurons ('attention_weighted', 'gradient_guided', 'rollout')
        
        Returns:
            Dictionary containing mamba neurons for each layer
        """
        mamba_neurons = {}
        
        for layer_idx, layer_data in attention_data.items():
            if method == 'attention_weighted':
                neurons = self._create_attention_weighted_neurons(layer_data)
            elif method == 'gradient_guided':
                neurons = self._create_gradient_guided_neurons(layer_data)
            elif method == 'rollout':
                neurons = self._create_rollout_neurons(layer_data)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            mamba_neurons[layer_idx] = neurons
        
        return mamba_neurons
    
    def _create_attention_weighted_neurons(self, layer_data: dict):
        """Create neurons weighted by attention vectors."""
        if 'attention_vectors' not in layer_data:
            return None
        
        attention_vectors = layer_data['attention_vectors']
        
        # Normalize attention vectors
        normalized_attention = (attention_vectors - attention_vectors.min()) / (attention_vectors.max() - attention_vectors.min())
        
        # Create neurons based on attention weights
        # Each neuron represents the weighted combination of input features
        neurons = {
            'attention_weights': normalized_attention,
            'neuron_activations': normalized_attention.mean(dim=-1),  # Average across sequence length
            'neuron_importance': normalized_attention.std(dim=-1)     # Variance as importance measure
        }
        
        return neurons
    
    def _create_gradient_guided_neurons(self, layer_data: dict):
        """Create neurons guided by gradients (requires gradient computation)."""
        if 'xai_vectors' not in layer_data:
            return None
        
        xai_vectors = layer_data['xai_vectors']
        
        # Create neurons based on xai vectors (cross-attention information)
        neurons = {
            'xai_vectors': xai_vectors,
            'neuron_activations': xai_vectors.mean(dim=-1),
            'neuron_importance': xai_vectors.std(dim=-1)
        }
        
        return neurons
    
    def _create_rollout_neurons(self, layer_data: dict):
        """Create neurons using rollout attention method."""
        if 'attention_vectors' not in layer_data:
            return None
        
        attention_vectors = layer_data['attention_vectors']
        
        # Apply rollout attention computation
        # This is similar to the rollout method in HiddenMambaAttn
        batch_size, num_heads, seq_len, _ = attention_vectors.shape
        
        # Create identity matrix for residual connections
        eye = torch.eye(seq_len).expand(batch_size, num_heads, seq_len, seq_len).to(attention_vectors.device)
        
        # Add residual connections and normalize
        attention_with_residual = attention_vectors + eye
        normalized_attention = attention_with_residual / attention_with_residual.sum(dim=-1, keepdim=True)
        
        # Create rollout neurons
        neurons = {
            'rollout_attention': normalized_attention,
            'neuron_activations': normalized_attention.mean(dim=1).mean(dim=-1),  # Average across heads and sequence
            'neuron_importance': normalized_attention.std(dim=1).mean(dim=-1)
        }
        
        return neurons
    
    def analyze_neuron_behavior(self, mamba_neurons: dict, layer_idx: int = 0):
        """
        Analyze the behavior of mamba neurons in a specific layer.
        
        Args:
            mamba_neurons: Dictionary containing mamba neurons
            layer_idx: Layer index to analyze
        
        Returns:
            Dictionary containing analysis results
        """
        if layer_idx not in mamba_neurons:
            return None
        
        neurons = mamba_neurons[layer_idx]
        
        analysis = {
            'layer_idx': layer_idx,
            'num_neurons': neurons['neuron_activations'].shape[-1] if 'neuron_activations' in neurons else 0,
            'mean_activation': neurons['neuron_activations'].mean().item() if 'neuron_activations' in neurons else 0,
            'activation_std': neurons['neuron_activations'].std().item() if 'neuron_activations' in neurons else 0,
            'top_neurons': None,
            'neuron_diversity': None
        }
        
        if 'neuron_activations' in neurons:
            activations = neurons['neuron_activations']
            # Find top neurons by activation
            top_indices = torch.argsort(activations, descending=True)[:10]
            analysis['top_neurons'] = [(idx.item(), activations[idx].item()) for idx in top_indices]
            
            # Calculate neuron diversity (entropy of activations)
            probs = torch.softmax(activations, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            analysis['neuron_diversity'] = entropy.item()
        
        return analysis
    
    def visualize_neurons(self, mamba_neurons: dict, layer_idx: Optional[int] = None, save_path: Optional[str] = None):
        """
        Visualize mamba neurons for all layers using the same graph structure.
        
        Args:
            mamba_neurons: Dictionary containing mamba neurons for all layers
            layer_idx: Optional layer index (ignored, plots all layers)
            save_path: Optional path to save the visualization
        """
        if not mamba_neurons:
            print("No neurons found")
            return
        
        # Get all layer indices
        layer_indices = sorted(mamba_neurons.keys())
        
        if not layer_indices:
            print("No layers found in mamba_neurons")
            return
        
        # Aggregate data from all layers
        all_activations = []
        all_importance = []
        all_attention_vectors = []
        
        for layer_idx in layer_indices:
            neurons = mamba_neurons[layer_idx]
            if 'neuron_activations' in neurons:
                activations = neurons['neuron_activations'].cpu().numpy()
                all_activations.append(activations)
            
            if 'neuron_importance' in neurons:
                importance = neurons['neuron_importance'].cpu().numpy()
                all_importance.append(importance)
            
            # Check for attention data in different possible keys
            if 'attention_weights' in neurons:
                attention = neurons['attention_weights'].cpu().numpy()
                all_attention_vectors.append(attention)
            elif 'attention_vectors' in neurons:
                attention = neurons['attention_vectors'].cpu().numpy()
                all_attention_vectors.append(attention)
        
        # Create the same 2x2 plot structure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Mamba Neurons Analysis - All Layers', fontsize=22, fontweight='bold')
        
        # Plot 1: Neuron activations (aggregated across all layers)
        if all_activations:
            # Flatten and combine all activations from all layers
            combined_activations = np.concatenate(all_activations)
            axes[0, 0].bar(range(len(combined_activations)), combined_activations)
            axes[0, 0].set_title('Neuron Activations (All Layers)', fontsize=18, fontweight='bold')
            axes[0, 0].set_xlabel('Neuron Index', fontsize=18)
            axes[0, 0].set_ylabel('Activation Value', fontsize=18)
            axes[0, 0].tick_params(labelsize=16)
        
        # Plot 2: Neuron importance (aggregated across all layers)
        if all_importance:
            # Flatten and combine all importance from all layers
            combined_importance = np.concatenate(all_importance)
            axes[0, 1].bar(range(len(combined_importance)), combined_importance)
            axes[0, 1].set_title('Neuron Importance (All Layers)', fontsize=18, fontweight='bold')
            axes[0, 1].set_xlabel('Neuron Index', fontsize=18)
            axes[0, 1].set_ylabel('Importance Score', fontsize=18)
            axes[0, 1].tick_params(labelsize=16)
        
        # Plot 3: Attention heatmap (average across all layers)
        if all_attention_vectors:
            # Average attention across all layers
            stacked_attention = np.stack([att.mean(axis=1) if len(att.shape) > 2 else att for att in all_attention_vectors], axis=0)
            avg_attention = np.mean(stacked_attention, axis=0)
            if len(avg_attention.shape) > 1:
                im = axes[1, 0].imshow(avg_attention[0] if len(avg_attention.shape) > 2 else avg_attention, cmap='viridis', aspect='auto')
                axes[1, 0].set_title('Attention Heatmap (Averaged Across Layers)', fontsize=18, fontweight='bold')
                axes[1, 0].set_xlabel('Sequence Position', fontsize=18)
                axes[1, 0].set_ylabel('Neuron Index', fontsize=18)
                axes[1, 0].tick_params(labelsize=16)
                cbar = plt.colorbar(im, ax=axes[1, 0])
                cbar.set_label('Attention Value', fontsize=18)
                cbar.ax.tick_params(labelsize=16)
        
        # Plot 4: Top neurons comparison (from all layers)
        if all_activations:
            combined_activations = np.concatenate(all_activations)
            top_indices = np.argsort(combined_activations)[-10:]  # Top 10 neurons
            top_activations = combined_activations[top_indices]
            axes[1, 1].bar(range(len(top_indices)), top_activations)
            axes[1, 1].set_title('Top 10 Neurons (All Layers)', fontsize=18, fontweight='bold')
            axes[1, 1].set_xlabel('Neuron Rank', fontsize=18)
            axes[1, 1].set_ylabel('Activation Value', fontsize=18)
            axes[1, 1].set_xticks(range(len(top_indices)))
            axes[1, 1].set_xticklabels([f'#{idx}' for idx in top_indices], fontsize=16)
            axes[1, 1].tick_params(labelsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.close()


def integrate_mamba_attention_neurons(model, inputs, layer_indices=None, methods=None):
    """
    Convenience function to integrate mamba attention neurons into existing analysis.
    
    Args:
        model: Mamba model
        inputs: Input tensor
        layer_indices: List of layer indices to analyze
        methods: List of methods to use for neuron creation
    
    Returns:
        Dictionary containing mamba neurons and analysis results
    """
    if methods is None:
        methods = ['attention_weighted', 'gradient_guided', 'rollout']
    
    if layer_indices is None:
        # Extract all layers by default
        layers = None
        if hasattr(model, 'layers'):
            layers = model.layers
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            layers = model.backbone.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        
        if layers is not None:
            layer_indices = list(range(len(layers)))
        else:
            layer_indices = [0, 6, 12, 18]  # Fallback default layers
    
    # Initialize the mamba attention neurons analyzer
    mamba_analyzer = MambaAttentionNeurons(model, enable_attention_computation=True)
    
    # Extract attention vectors
    print("Extracting attention vectors from Mamba model...")
    attention_data = mamba_analyzer.extract_attention_vectors(inputs, layer_indices)
    
    # Create mamba neurons using different methods
    mamba_neurons = {}
    for method in methods:
        print(f"Creating mamba neurons using {method} method...")
        neurons = mamba_analyzer.create_mamba_neurons(attention_data, method)
        mamba_neurons[method] = neurons
    
    # Analyze neuron behavior
    analysis_results = {}
    for method in methods:
        if method in mamba_neurons:
            analysis_results[method] = {}
            for layer_idx in layer_indices:
                if layer_idx in mamba_neurons[method]:
                    analysis = mamba_analyzer.analyze_neuron_behavior(mamba_neurons[method], layer_idx)
                    if analysis:
                        analysis_results[method][layer_idx] = analysis
    
    return {
        'attention_data': attention_data,
        'mamba_neurons': mamba_neurons,
        'analysis_results': analysis_results,
        'analyzer': mamba_analyzer
    }
