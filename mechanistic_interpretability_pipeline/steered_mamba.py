"""
Steered Mamba Implementation

This module provides steering functionality for Mamba models using Cluster 9 neurons.
Steering allows for targeted activation manipulation to influence model behavior.
"""

import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class SimpleSteering:
    """
    Simple steering using Cluster 9 neurons (proven approach).
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
        # Try to find layers in the model
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            self.layers = model.backbone.layers
            logger.info(f"Found {len(self.layers)} layers in model.backbone.layers")
        elif hasattr(model, 'layers'):
            self.layers = model.layers
            logger.info(f"Found {len(self.layers)} layers in model.layers")
        else:
            logger.error("Could not find layers in model! Available attributes: " + str(dir(model)))
            raise AttributeError("Model does not have 'backbone.layers' or 'layers' attribute")
        
        # Cluster 9 neurons from original research
        self.cluster9_neurons = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
        logger.info(f"Initialized SimpleSteering with {len(self.cluster9_neurons)} Cluster 9 neurons")
    
    def apply_steering(self, strength: float = 5.0, layer_idx: int = 20):
        """
        Apply Cluster 9 steering at specified layer.
        
        Args:
            strength: Multiplicative factor to apply to Cluster 9 neurons
            layer_idx: Index of the layer to apply steering to
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"🎯 Applying steering: Layer {layer_idx}, Strength {strength}x")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        if target is layer:
            logger.warning(f"⚠️ Could not find 'mixer' or 'ssm' in layer {layer_idx}, using layer directly")
            logger.debug(f"   Layer attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            h_mod = hidden.clone()
            for idx in self.cluster9_neurons:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def remove_steering(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def apply_multi_layer_steering(self, layer_indices: List[int], strength: float = 5.0):
        """
        Apply steering to multiple layers.
        
        Args:
            layer_indices: List of layer indices to apply steering to
            strength: Multiplicative factor to apply to Cluster 9 neurons
        """
        for layer_idx in layer_indices:
            self.apply_steering(strength=strength, layer_idx=layer_idx)

