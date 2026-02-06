#!/usr/bin/env python3
"""
Create and save a steered mamba model for analysis.
This applies steering hooks and saves the model state.
"""

import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer
from mamba_model_loader import load_mamba_model_and_tokenizer
from steered_mamba import SimpleSteering

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class SteeredMambaModel(torch.nn.Module):
    """
    Wrapper that applies steering to a Mamba model.
    This ensures steering is always active when the model is used.
    """
    
    def __init__(self, base_model, steering, layer_indices=None, strength=5.0):
        super().__init__()
        self.base_model = base_model
        self.steering = steering
        self.layer_indices = layer_indices or [0, 6, 12, 18]
        self.strength = strength
        
        # Apply steering immediately
        logger.info(f"🎯 Applying steering to layers {self.layer_indices} with strength {strength}x")
        self.steering.apply_multi_layer_steering(self.layer_indices, strength=strength)
        logger.info(f"✅ Steering applied: {len(self.steering.hooks)} hooks registered")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model with steering active."""
        return self.base_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    def remove_steering(self):
        """Remove steering hooks."""
        self.steering.remove_steering()
    
    def reapply_steering(self):
        """Reapply steering hooks."""
        self.steering.remove_steering()
        self.steering.apply_multi_layer_steering(self.layer_indices, strength=self.strength)


def create_and_save_steered_model(
    model_name="state-spaces/mamba-130m-hf",
    output_dir="steered_models",
    layer_indices=[0, 6, 12, 18],
    strength=5.0,
    device="cuda"
):
    """
    Create a steered mamba model and save it.
    
    Args:
        model_name: Name of the model to load
        output_dir: Directory to save the steered model
        layer_indices: Layers to apply steering to
        strength: Steering strength
        device: Device to use
    """
    logger.info("=" * 80)
    logger.info("Creating Steered Mamba Model")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Load base model
    logger.info(f"\n1. Loading base model: {model_name}")
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name=model_name,
        device=device,
        use_mamba_class=True,
        fallback_to_auto=True
    )
    logger.info("✅ Base model loaded")
    
    # Initialize steering
    logger.info(f"\n2. Initializing steering...")
    steering = SimpleSteering(model)
    logger.info(f"✅ Steering initialized")
    logger.info(f"   - Layers: {len(steering.layers)}")
    logger.info(f"   - Cluster 9 neurons: {len(steering.cluster9_neurons)}")
    
    # Create steered model wrapper
    logger.info(f"\n3. Creating steered model wrapper...")
    steered_model = SteeredMambaModel(
        base_model=model,
        steering=steering,
        layer_indices=layer_indices,
        strength=strength
    )
    logger.info("✅ Steered model wrapper created")
    
    # Verify steering is active
    logger.info(f"\n4. Verifying steering is active...")
    test_text = "The quick brown fox"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = steered_model(**inputs)
        logits = output.logits
    
    logger.info(f"✅ Model forward pass successful")
    logger.info(f"   Output shape: {logits.shape}")
    logger.info(f"   Active hooks: {len(steering.hooks)}")
    
    # Save model and tokenizer
    logger.info(f"\n5. Saving steered model...")
    model_save_path = output_path / "steered_mamba_model.pt"
    tokenizer_save_path = output_path / "tokenizer"
    config_save_path = output_path / "steering_config.json"
    
    # Save model state (without hooks, but we'll save the config)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config.__dict__ if hasattr(model, 'config') else {},
        'layer_indices': layer_indices,
        'strength': strength,
        'model_name': model_name,
    }, model_save_path)
    logger.info(f"✅ Model state saved to: {model_save_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_save_path)
    logger.info(f"✅ Tokenizer saved to: {tokenizer_save_path}")
    
    # Save steering config
    import json
    steering_config = {
        'layer_indices': layer_indices,
        'strength': strength,
        'cluster9_neurons': steering.cluster9_neurons,
        'num_layers': len(steering.layers),
        'model_name': model_name,
    }
    with open(config_save_path, 'w') as f:
        json.dump(steering_config, f, indent=2)
    logger.info(f"✅ Steering config saved to: {config_save_path}")
    
    # Save a marker file indicating this is a steered model
    marker_path = output_path / "STEERED_MODEL.txt"
    with open(marker_path, 'w') as f:
        f.write(f"Steered Mamba Model\n")
        f.write(f"==================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Steering layers: {layer_indices}\n")
        f.write(f"Steering strength: {strength}x\n")
        f.write(f"Cluster 9 neurons: {len(steering.cluster9_neurons)}\n")
        f.write(f"\nThis model has steering hooks applied.\n")
        f.write(f"Load using: load_steered_model('{output_path}')\n")
    logger.info(f"✅ Marker file saved to: {marker_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Steered model creation complete!")
    logger.info(f"   Model saved to: {model_save_path}")
    logger.info(f"   Tokenizer saved to: {tokenizer_save_path}")
    logger.info(f"   Config saved to: {config_save_path}")
    logger.info(f"\n   To use this model, load it with:")
    logger.info(f"   from create_steered_model import load_steered_model")
    logger.info(f"   model, tokenizer = load_steered_model('{output_path}')")
    logger.info("=" * 80)
    
    return steered_model, tokenizer, output_path


def load_steered_model(model_dir, device="cuda"):
    """
    Load a previously saved steered model.
    
    Args:
        model_dir: Directory containing the saved steered model
        device: Device to load the model on
    
    Returns:
        (steered_model, tokenizer, steering_config)
    """
    model_dir = Path(model_dir)
    
    logger.info(f"Loading steered model from: {model_dir}")
    
    # Load config
    config_path = model_dir / "steering_config.json"
    import json
    with open(config_path, 'r') as f:
        steering_config = json.load(f)
    
    logger.info(f"   Steering config: layers {steering_config['layer_indices']}, strength {steering_config['strength']}x")
    
    # Load base model
    model_name = steering_config.get('model_name', 'state-spaces/mamba-130m-hf')
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name=model_name,
        device=device,
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    # Load model state if available
    model_path = model_dir / "steered_mamba_model.pt"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Loaded model state from checkpoint")
    
    # Initialize steering and create wrapper
    steering = SimpleSteering(model)
    steered_model = SteeredMambaModel(
        base_model=model,
        steering=steering,
        layer_indices=steering_config['layer_indices'],
        strength=steering_config['strength']
    )
    
    logger.info("✅ Steered model loaded successfully")
    logger.info(f"   Active hooks: {len(steering.hooks)}")
    
    return steered_model, tokenizer, steering_config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and save a steered mamba model")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf",
                       help="Model name to load")
    parser.add_argument("--output", type=str, default="steered_models",
                       help="Output directory for steered model")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 6, 12, 18],
                       help="Layer indices to apply steering to")
    parser.add_argument("--strength", type=float, default=5.0,
                       help="Steering strength")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    create_and_save_steered_model(
        model_name=args.model,
        output_dir=args.output,
        layer_indices=args.layers,
        strength=args.strength,
        device=args.device
    )

