"""
Recursive Pattern Visualizer for Mamba Models

This module creates visualizations to understand how Mamba's recursive properties
affect successive layers and how activations evolve through the model.

Key visualizations:
- Layer-wise activation heatmaps
- Recursive state evolution plots
- Cross-layer correlation matrices
- Temporal autocorrelation patterns
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ Matplotlib not available - visualization disabled")

from ssm_component_extractor import SSMComponentExtractor
from layer_correlation_analyzer import LayerCorrelationAnalyzer


class RecursiveVisualizer:
    """
    Creates visualizations for understanding Mamba's recursive properties.
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        
        # Initialize analyzers
        self.ssm_extractor = SSMComponentExtractor(model, self.device)
        self.correlation_analyzer = LayerCorrelationAnalyzer(model, self.device)
        
        # Create output directory
        os.makedirs("recursive_analysis_plots", exist_ok=True)
        
    def visualize_layer_activations(self, layer_indices: List[int], input_text: str, 
                                  save_prefix: str = "layer_activations") -> Dict:
        """
        Visualize activations across multiple layers.
        
        Args:
            layer_indices: List of layer indices to visualize
            input_text: Input text to process
            save_prefix: Prefix for saved files
        
        Returns:
            Dictionary containing visualization data
        """
        if not HAS_PLOTTING:
            print("⚠️ Matplotlib not available - skipping visualization")
            return {}
        
        print(f"🎨 Visualizing layer activations for layers: {layer_indices}")
        
        # Extract activations
        activations = self.correlation_analyzer.extract_layer_activations(layer_indices, input_text)
        
        # Create subplot grid
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 10))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Layer Activations Analysis\nInput: "{input_text[:50]}..."', fontsize=30, fontweight='bold')
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx not in activations:
                continue
                
            activation = activations[layer_idx]  # [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activation.shape
            
            # Take first batch and transpose for visualization
            h = activation[0].cpu().numpy()  # [seq_len, hidden_dim]
            
            # Plot 1: Activation heatmap
            im1 = axes[0, i].imshow(h.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
            axes[0, i].set_title(f'Layer {layer_idx} Activations', fontsize=27, fontweight='bold')
            axes[0, i].set_xlabel('Time Step', fontsize=27, fontweight='bold')
            axes[0, i].set_ylabel('Hidden Dimension', fontsize=27, fontweight='bold')
            axes[0, i].tick_params(axis='both', which='major', labelsize=24)
            axes[0, i].tick_params(axis='both', which='minor', labelsize=21)
            plt.colorbar(im1, ax=axes[0, i])
            
            # Plot 2: Activation magnitude over time
            magnitudes = np.linalg.norm(h, axis=1)  # [seq_len]
            axes[1, i].plot(magnitudes, 'b-', linewidth=2, label='Magnitude')
            axes[1, i].set_title(f'Layer {layer_idx} State Magnitude', fontsize=27, fontweight='bold')
            axes[1, i].set_xlabel('Time Step', fontsize=27, fontweight='bold')
            axes[1, i].set_ylabel('Activation Magnitude', fontsize=27, fontweight='bold')
            axes[1, i].tick_params(axis='both', which='major', labelsize=24)
            axes[1, i].tick_params(axis='both', which='minor', labelsize=21)
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend(fontsize=24, prop={'weight': 'bold'})
            
            # Add statistics text
            stats_text = f'Mean: {magnitudes.mean():.4f}\nStd: {magnitudes.std():.4f}'
            axes[1, i].text(0.02, 0.98, stats_text, transform=axes[1, i].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=25, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"recursive_analysis_plots/{save_prefix}_layers_{'_'.join(map(str, layer_indices))}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Layer activations plot saved to: {save_path}")
        
        plt.close()
        return {'save_path': save_path, 'layer_indices': layer_indices}
    
    def visualize_cross_layer_correlations(self, layer_indices: List[int], input_text: str,
                                         save_prefix: str = "cross_layer_correlations") -> Dict:
        """
        Visualize correlations between layers.
        
        Args:
            layer_indices: List of layer indices to analyze
            input_text: Input text to process
            save_prefix: Prefix for saved files
        
        Returns:
            Dictionary containing correlation visualization data
        """
        if not HAS_PLOTTING:
            print("⚠️ Matplotlib not available - skipping visualization")
            return {}
        
        print(f"🔗 Visualizing cross-layer correlations for layers: {layer_indices}")
        
        # Extract activations and compute correlations
        activations = self.correlation_analyzer.extract_layer_activations(layer_indices, input_text)
        correlations = self.correlation_analyzer.compute_cross_layer_correlations()
        
        # Create subplot grid
        n_pairs = len(correlations)
        if n_pairs == 0:
            print("⚠️ No correlation pairs found")
            return {}
        
        fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 4.5))
        if n_pairs == 1:
            axes = [axes]
        
        fig.suptitle(f'Cross-Layer Correlations\nInput: "{input_text[:50]}..."', fontsize=14, fontweight='bold')
        
        for i, (pair_key, data) in enumerate(correlations.items()):
            corr_matrix = data['correlation_matrix'].cpu().numpy()
            
            # Plot correlation matrix
            im = axes[i].imshow(corr_matrix, aspect='auto', cmap='RdBu_r', 
                               vmin=-1, vmax=1, interpolation='nearest')
            axes[i].set_title(f'{pair_key}\nMean: {data["mean_correlation"]:.4f}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Layer j Dimensions', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Layer i\nDimensions', fontsize=14, fontweight='bold')
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            axes[i].tick_params(axis='both', which='minor', labelsize=14)
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"recursive_analysis_plots/{save_prefix}_layers_{'_'.join(map(str, layer_indices))}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Cross-layer correlations plot saved to: {save_path}")
        
        plt.close()
        return {'save_path': save_path, 'correlations': correlations}
    
    def visualize_recursive_patterns(self, layer_indices: List[int], input_text: str,
                                   save_prefix: str = "recursive_patterns") -> Dict:
        """
        Visualize recursive patterns within layers.
        
        Args:
            layer_indices: List of layer indices to analyze
            input_text: Input text to process
            save_prefix: Prefix for saved files
        
        Returns:
            Dictionary containing recursive pattern visualization data
        """
        if not HAS_PLOTTING:
            print("⚠️ Matplotlib not available - skipping visualization")
            return {}
        
        print(f"🔄 Visualizing recursive patterns for layers: {layer_indices}")
        
        # Extract activations and analyze patterns
        activations = self.correlation_analyzer.extract_layer_activations(layer_indices, input_text)
        
        # Analyze recursive patterns for each layer
        patterns = {}
        for layer_idx in layer_indices:
            if layer_idx in activations:
                patterns[layer_idx] = self.correlation_analyzer.analyze_recursive_patterns(layer_idx)
        
        # Create subplot grid
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(2, n_layers, figsize=(6*n_layers, 10))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Recursive Patterns Analysis\nInput: "{input_text[:50]}..."', fontsize=30, fontweight='bold')
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx not in patterns:
                continue
            
            pattern = patterns[layer_idx]
            activation = activations[layer_idx][0].cpu().numpy()  # [seq_len, hidden_dim]
            
            # Plot 1: Temporal autocorrelation
            if 'temporal_autocorrelation' in pattern:
                autocorr = pattern['temporal_autocorrelation']
                lags = [int(k.split('_')[1]) for k in autocorr.keys()]
                means = [autocorr[f'lag_{lag}']['mean'] for lag in lags]
                
                axes[0, i].plot(lags, means, 'bo-', linewidth=2, markersize=6)
                axes[0, i].set_title(f'Layer {layer_idx} Temporal Autocorrelation', fontsize=27, fontweight='bold')
                axes[0, i].set_xlabel('Lag', fontsize=27, fontweight='bold')
                axes[0, i].set_ylabel('Correlation', fontsize=27, fontweight='bold')
                axes[0, i].tick_params(axis='both', which='major', labelsize=24)
                axes[0, i].tick_params(axis='both', which='minor', labelsize=21)
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # Plot 2: State evolution
            if 'state_evolution' in pattern:
                evolution = pattern['state_evolution']
                magnitudes = np.linalg.norm(activation, axis=1)
                
                axes[1, i].plot(magnitudes, 'g-', linewidth=2, label='State Magnitude')
                axes[1, i].set_title(f'Layer {layer_idx} State Evolution', fontsize=27, fontweight='bold')
                axes[1, i].set_xlabel('Time Step', fontsize=27, fontweight='bold')
                axes[1, i].set_ylabel('State Magnitude', fontsize=27, fontweight='bold')
                axes[1, i].tick_params(axis='both', which='major', labelsize=24)
                axes[1, i].tick_params(axis='both', which='minor', labelsize=21)
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].legend(fontsize=24, prop={'weight': 'bold'})
                
                # Add trend information
                trend = evolution['state_magnitude']['trend']
                axes[1, i].text(0.02, 0.98, f'Trend: {trend}', 
                               transform=axes[1, i].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                               fontsize=25, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"recursive_analysis_plots/{save_prefix}_layers_{'_'.join(map(str, layer_indices))}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Recursive patterns plot saved to: {save_path}")
        
        plt.close()
        return {'save_path': save_path, 'patterns': patterns}
    
    def visualize_ssm_components(self, layer_indices: List[int], input_text: str,
                               save_prefix: str = "ssm_components") -> Dict:
        """
        Visualize SSM components across layers.
        
        Args:
            layer_indices: List of layer indices to analyze
            input_text: Input text to process
            save_prefix: Prefix for saved files
        
        Returns:
            Dictionary containing SSM component visualization data
        """
        if not HAS_PLOTTING:
            print("⚠️ Matplotlib not available - skipping visualization")
            return {}
        
        print(f"🧠 Visualizing SSM components for layers: {layer_indices}")
        
        # Extract SSM components
        ssm_components = self.ssm_extractor.extract_ssm_components(layer_indices, input_text)
        
        # Create subplot grid
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(2, n_layers, figsize=(6*n_layers, 10))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'SSM Components Analysis\nInput: "{input_text[:50]}..."', fontsize=14, fontweight='bold')
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx not in ssm_components:
                continue
            
            components = ssm_components[layer_idx]
            
            # Plot 1: A matrix (if available)
            if components['A_matrix'] is not None:
                A = components['A_matrix'].cpu().numpy()
                if A.ndim > 2:
                    A = A.reshape(A.shape[-2], A.shape[-1])
                
                im1 = axes[0, i].imshow(A, aspect='auto', cmap='RdBu_r', interpolation='nearest')
                axes[0, i].set_title(f'Layer {layer_idx} A Matrix\nShape: {A.shape}', fontsize=14, fontweight='bold')
                axes[0, i].set_xlabel('State Dimension', fontsize=14, fontweight='bold')
                axes[0, i].set_ylabel('Hidden Dimension', fontsize=14, fontweight='bold')
                axes[0, i].tick_params(axis='both', which='major', labelsize=14)
                axes[0, i].tick_params(axis='both', which='minor', labelsize=14)
                plt.colorbar(im1, ax=axes[0, i])
            else:
                axes[0, i].text(0.5, 0.5, 'A Matrix\nNot Available', 
                               ha='center', va='center', transform=axes[0, i].transAxes,
                               fontsize=14, fontweight='bold')
                axes[0, i].set_title(f'Layer {layer_idx} A Matrix', fontsize=14, fontweight='bold')
            
            # Plot 2: Hidden state evolution
            if components['hidden_states'] is not None:
                h = components['hidden_states'][0].cpu().numpy()  # [seq_len, hidden_dim]
                
                # Plot first few dimensions
                for j in range(min(5, h.shape[1])):
                    axes[1, i].plot(h[:, j], label=f'Dim {j}', alpha=0.7)
                
                axes[1, i].set_title(f'Layer {layer_idx} Hidden State Evolution', fontsize=14, fontweight='bold')
                axes[1, i].set_xlabel('Time Step', fontsize=14, fontweight='bold')
                axes[1, i].set_ylabel('Activation Value', fontsize=14, fontweight='bold')
                axes[1, i].tick_params(axis='both', which='major', labelsize=14)
                axes[1, i].tick_params(axis='both', which='minor', labelsize=14)
                axes[1, i].legend(fontsize=14, prop={'weight': 'bold'})
                axes[1, i].grid(True, alpha=0.3)
            else:
                axes[1, i].text(0.5, 0.5, 'Hidden States\nNot Available', 
                               ha='center', va='center', transform=axes[1, i].transAxes,
                               fontsize=14, fontweight='bold')
                axes[1, i].set_title(f'Layer {layer_idx} Hidden State Evolution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"recursive_analysis_plots/{save_prefix}_layers_{'_'.join(map(str, layer_indices))}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 SSM components plot saved to: {save_path}")
        
        plt.close()
        return {'save_path': save_path, 'ssm_components': ssm_components}
    
    def create_comprehensive_analysis(self, layer_indices: List[int], input_text: str) -> Dict:
        """
        Create a comprehensive analysis with all visualizations.
        
        Args:
            layer_indices: List of layer indices to analyze
            input_text: Input text to process
        
        Returns:
            Dictionary containing all analysis results
        """
        print(f"🎯 Creating comprehensive recursive analysis for layers: {layer_indices}")
        print(f"📝 Input: '{input_text}'")
        
        results = {
            'input_text': input_text,
            'layer_indices': layer_indices,
            'visualizations': {}
        }
        
        # 1. Layer activations visualization
        print("\n1️⃣ Creating layer activations visualization...")
        results['visualizations']['layer_activations'] = self.visualize_layer_activations(
            layer_indices, input_text
        )
        
        # 2. Cross-layer correlations visualization
        print("\n2️⃣ Creating cross-layer correlations visualization...")
        results['visualizations']['cross_layer_correlations'] = self.visualize_cross_layer_correlations(
            layer_indices, input_text
        )
        
        # 3. Recursive patterns visualization
        print("\n3️⃣ Creating recursive patterns visualization...")
        results['visualizations']['recursive_patterns'] = self.visualize_recursive_patterns(
            layer_indices, input_text
        )
        
        # 4. SSM components visualization
        print("\n4️⃣ Creating SSM components visualization...")
        results['visualizations']['ssm_components'] = self.visualize_ssm_components(
            layer_indices, input_text
        )
        
        # 5. Generate analysis report
        print("\n5️⃣ Generating analysis report...")
        correlation_report = self.correlation_analyzer.generate_analysis_report()
        results['analysis_report'] = correlation_report
        
        # Save comprehensive results
        with open('recursive_analysis_plots/comprehensive_analysis.json', 'w') as f:
            def convert_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tensors(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_tensors(results), f, indent=2)
        
        print(f"\n💾 Comprehensive analysis saved to: recursive_analysis_plots/comprehensive_analysis.json")
        print(f"📁 All plots saved in: recursive_analysis_plots/")
        
        return results


def demonstrate_recursive_visualization():
    """Demonstrate recursive pattern visualization on a Mamba model."""
    print("🚀 Recursive Pattern Visualization Demo")
    print("=" * 60)
    
    # Load model
    from transformers import AutoModelForCausalLM
    model_name = "state-spaces/mamba-130m-hf"
    print(f"📥 Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize visualizer
    visualizer = RecursiveVisualizer(model, device)
    
    # Test input
    test_text = "Mamba models use recursive state space models to process sequences efficiently through selective state updates."
    print(f"📝 Test input: '{test_text}'")
    
    # Create comprehensive analysis - use all layers
    try:
        num_layers = visualizer.model.config.num_hidden_layers
        layer_indices = list(range(num_layers))  # All layers
    except:
        layer_indices = list(range(24))  # Default to 24 layers
    results = visualizer.create_comprehensive_analysis(layer_indices, test_text)
    
    print("\n" + "="*60)
    print("📊 VISUALIZATION SUMMARY")
    print("="*60)
    
    print(f"\n🎨 Generated visualizations:")
    for viz_type, viz_data in results['visualizations'].items():
        if 'save_path' in viz_data:
            print(f"  ✅ {viz_type}: {viz_data['save_path']}")
    
    print(f"\n📈 Analysis summary:")
    if 'analysis_report' in results:
        report = results['analysis_report']
        print(f"  Layers analyzed: {report['summary']['num_layers_analyzed']}")
        print(f"  Correlation pairs: {report['summary']['correlation_analysis']['num_pairs']}")
        print(f"  Mean correlation: {report['summary']['correlation_analysis']['mean_correlation']:.4f}")
    
    print(f"\n✅ Recursive visualization analysis complete!")
    print(f"📁 Check the 'recursive_analysis_plots/' directory for all generated plots")
    
    return visualizer, results


if __name__ == "__main__":
    demonstrate_recursive_visualization()
