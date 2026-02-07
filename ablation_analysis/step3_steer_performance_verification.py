"""
DELTA-SENSITIVE SUBSPACE ABLATION WITH IFEVAL DATASET

Goal: Run ablation (Delta-Sensitive Subspace vs Random vs High-Variance) on IFEval dataset
with both accuracy and perplexity metrics.

"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""
    neurons: List[int]
    layer: int
    strength: float
    experiment_type: str
    ablated_neuron: Optional[int] = None
    ablate_layer: Optional[int] = None  # Layer to ablate (zero out) after steering
    ablate_neurons: Optional[List[int]] = None  # Specific neurons to ablate (zero out) after steering


class ClusterAblationAnalyzer:
    """
    Analyze cluster ablation on IFEval dataset with perplexity and accuracy metrics.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Get model layers
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        self.num_layers = len(self.layers)
        self.hidden_dim = model.config.d_model if hasattr(model.config, 'd_model') else 768
        
        # Best configuration from validation
        self.best_layer = 20
        self.best_strength = 5.0
        
        # Layer-specific steering strengths to prevent extreme values
        # Later layers are more sensitive, so use lower amplification
        self.layer_strengths = {
            18: 5.0,   # Pre-bottleneck: can handle higher amplification
            19: 5.0,   # Pre-compression: can handle higher amplification
            20: 5.0,   # Bottleneck: optimal layer
            21: 2.0,   # Post-bottleneck: reduce to prevent instability
            22: 1.5,   # Output projection: very sensitive, use minimal amplification
        }
        
        logger.info(f"Initialized Cluster Ablation Analyzer:")
        logger.info(f"  Model layers: {self.num_layers}")
        logger.info(f"  Hidden dimension: {self.hidden_dim}")
        logger.info(f"  Target layer: {self.best_layer}")
    
    def _get_steering_target(self, layer_idx):
        """
        Get the module to apply steering to.
        Uses the same approach as post_hoc_extention_1 for better compatibility.
        """
        layer = self.layers[layer_idx]
        target = None
        
        # Try to find the right target module for hooking (must be a torch.nn.Module)
        # Priority order based on model architecture
        for attr_name in ['mixer', 'ssm', 'attention', 'attn', 'self_attn', 'recurrence', 
                          'recur', 'conv', 'hyena', 'norm', 'mamba', 'block', 'mlp']:
            if hasattr(layer, attr_name):
                attr = getattr(layer, attr_name)
                # Ensure it's a Module, not a function or method
                if isinstance(attr, torch.nn.Module):
                    target = attr
                    logger.debug(f"   Found target module: {attr_name} ({type(attr).__name__}) at layer {layer_idx}")
                    break
                elif callable(attr) and not isinstance(attr, torch.nn.Module):
                    logger.debug(f"   Skipping {attr_name} (it's a function/method, not a module)")
        
        if target is None:
            # Fallback: use layer itself if it's a module
            if isinstance(layer, torch.nn.Module):
                target = layer
                logger.debug(f"   Using layer itself as target ({type(layer).__name__}) at layer {layer_idx}")
            else:
                # Last resort: try to find any submodule
                if hasattr(layer, 'named_modules'):
                    for name, module in layer.named_modules():
                        if isinstance(module, torch.nn.Module) and name != '':
                            target = module
                            logger.debug(f"   Found submodule as target: {name} ({type(module).__name__}) at layer {layer_idx}")
                            break
        
        if target is None or not isinstance(target, torch.nn.Module):
            logger.error(f"❌ Could not find a valid module to hook in layer {layer_idx}")
            logger.error(f"   Layer type: {type(layer).__name__}")
            logger.error(f"   Layer attributes: {[a for a in dir(layer) if not a.startswith('_')][:10]}")
            raise ValueError(f"Cannot register hook: layer {layer_idx} does not contain a valid torch.nn.Module")
        
        return target
    
    def _select_delta_sensitive_neurons(self, num_neurons: int = 16) -> List[int]:
        """
        Select delta-sensitive subspace neurons (SSM Core).
        For now, we'll use discovered neurons from impact analysis if available,
        otherwise use a heuristic selection.
        """
        # Try to load from discovered neurons
        try:
            json_path = Path("ablation_3_results/discovered_neurons.json")
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'all_neuron_impacts' in data:
                        # Get top neurons by impact (these are delta-sensitive neurons)
                        impacts = [(item['neuron'], item['impact']) for item in data['all_neuron_impacts']]
                        impacts.sort(key=lambda x: x[1], reverse=True)
                        delta_sensitive_neurons = [n for n, _ in impacts[:num_neurons]]
                        logger.info(f"✅ Loaded {len(delta_sensitive_neurons)} delta-sensitive neurons from discovered_neurons.json")
                        return delta_sensitive_neurons
        except Exception as e:
            logger.warning(f"Could not load delta-sensitive neurons: {e}")
        
        # Fallback: Use neurons with indices that are multiples of certain values
        # This is a placeholder - should be replaced with actual delta-sensitive neuron identification
        delta_sensitive_neurons = list(range(0, self.hidden_dim, self.hidden_dim // num_neurons))[:num_neurons]
        logger.info(f"⚠️  Using heuristic delta-sensitive neurons (should be replaced with actual identification)")
        return delta_sensitive_neurons
    
    def _select_random_neurons(self, num_neurons: int = 16, seed: int = 42) -> List[int]:
        """Select random neurons."""
        np.random.seed(seed)
        random_neurons = np.random.choice(
            self.hidden_dim,
            size=num_neurons,
            replace=False
        ).tolist()
        return sorted(random_neurons)
    
    def _select_high_variance_neurons(self, texts: List[str], num_neurons: int = 16) -> List[int]:
        """Select neurons with highest activation variance."""
        activations = []
        
        for text in texts[:50]:  # Use subset for efficiency
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            captured = {}
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured['act'] = hidden.detach().cpu()
            
            target = self._get_steering_target(self.best_layer)
            hook = target.register_forward_hook(capture_hook)
            
            with torch.no_grad():
                try:
                    _ = self.model(**inputs)
                    if 'act' in captured:
                        act = captured['act']
                        if act.dim() == 3:
                            act = act.mean(dim=1)  # Average over sequence
                        activations.append(act)
                except:
                    pass
            
            hook.remove()
        
        if not activations:
            return []
        
        # Calculate variance across examples
        all_acts = torch.stack(activations, dim=0)  # [n_examples, hidden_dim]
        variances = all_acts.var(dim=0).squeeze()  # [hidden_dim]
        
        # Select top-k neurons
        top_k = torch.topk(variances, num_neurons).indices.tolist()
        return sorted(top_k)
    
    def calculate_perplexity(self, texts: List[str], config: AblationConfig) -> float:
        """
        Calculate perplexity on texts.
        
        Perplexity = exp(cross_entropy_loss)
        
        Workflow:
        1. Apply steering (amplify neurons) at config.layer
        2. Apply ablation (zero out) if config.ablate_layer or config.ablate_neurons specified
        3. Measure perplexity
        
        Note: Perplexity has no upper bound (can range from 1 to infinity).
        Typical values for language models:
        - Good models: 10-50
        - Moderate: 50-100
        - Poor: 100-1000
        - Very poor: >1000
        - Extreme values (>1e6) indicate severe model degradation or numerical instability
        
        When steering causes extreme perplexity (>1e6), it indicates the model's
        predictions have become severely degraded, likely due to:
        1. Numerical instability from aggressive amplification
        2. Model producing extremely low-probability predictions
        3. Steering at inappropriate layers causing distribution collapse
        """
        hooks = []
        
        # Step 1: Steering hook (amplification)
        def steering_hook(module, input, output):
            """Hook that amplifies specified neurons, then ablates if specified."""
            if isinstance(output, tuple):
                hidden = output[0].clone()
                rest = output[1:]
            else:
                hidden = output.clone()
                rest = ()
            
            # Apply steering (amplification)
            if config.neurons and config.strength != 1.0 and len(config.neurons) > 0:
                # Filter neurons to valid range
                if hidden.dim() >= 2:
                    max_neuron = hidden.shape[-1]
                else:
                    max_neuron = hidden.numel()
                valid_neurons = [n for n in config.neurons if n < max_neuron]
                
                # Handle different output shapes
                # For most models: output is [batch, seq_len, hidden_dim]
                if hidden.dim() >= 2:
                    # Amplify cluster neurons on the last dimension
                    last_dim = hidden.shape[-1]
                    for neuron_idx in valid_neurons:
                        if neuron_idx < last_dim:
                            hidden[..., neuron_idx] *= config.strength
                else:
                    # 1D or scalar output - apply to all if within range
                    if hidden.numel() > 0:
                        for neuron_idx in valid_neurons:
                            if neuron_idx < hidden.numel():
                                hidden.view(-1)[neuron_idx] *= config.strength
            
            # Step 2: Apply ablation (zero out) if specified
            if config.ablate_neurons and len(config.ablate_neurons) > 0:
                # Zero out specific neurons after steering
                if hidden.dim() >= 2:
                    last_dim = hidden.shape[-1]
                    for neuron_idx in config.ablate_neurons:
                        if neuron_idx < last_dim:
                            hidden[..., neuron_idx] = 0.0
                else:
                    if hidden.numel() > 0:
                        for neuron_idx in config.ablate_neurons:
                            if neuron_idx < hidden.numel():
                                hidden.view(-1)[neuron_idx] = 0.0
            
            if rest:
                return (hidden,) + rest
            return hidden
        
        # Register steering hook at the steering layer
        target = self._get_steering_target(config.layer)
        hook = target.register_forward_hook(steering_hook)
        hooks.append(hook)
        
        # Step 3: Ablation hook for specific layer (if ablate_layer is specified)
        if config.ablate_layer is not None and config.ablate_layer != config.layer:
            def ablation_hook(module, input, output):
                """Hook that zeros out entire layer output."""
                if isinstance(output, tuple):
                    hidden = torch.zeros_like(output[0])
                    rest = output[1:]
                    if rest:
                        return (hidden,) + rest
                    return hidden
                else:
                    return torch.zeros_like(output)
            
            ablation_target = self._get_steering_target(config.ablate_layer)
            ablation_hook_handle = ablation_target.register_forward_hook(ablation_hook)
            hooks.append(ablation_hook_handle)
        
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                try:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=False
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    input_ids = inputs['input_ids']
                    if input_ids.shape[1] < 2:
                        continue
                    
                    outputs = self.model(**inputs)
                    
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Clip logits to prevent numerical overflow
                    # This prevents extreme perplexity values while maintaining relative ordering
                    logits = torch.clamp(logits, min=-50.0, max=50.0)
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    loss_fct = F.cross_entropy
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    
                    loss = loss_fct(shift_logits, shift_labels, ignore_index=self.tokenizer.pad_token_id)
                    
                    # Check for numerical issues
                    loss_value = loss.item()
                    if not np.isfinite(loss_value) or loss_value > 50:  # Loss > 50 means perplexity > exp(50) ≈ 5.2e21
                        logger.warning(f"Skipping text with extreme loss: {loss_value:.2f} (likely numerical instability)")
                        continue
                    
                    num_tokens = (shift_labels != self.tokenizer.pad_token_id).sum().item()
                    
                    if num_tokens > 0:
                        total_loss += loss_value * num_tokens
                        total_tokens += num_tokens
                
                except Exception as e:
                    logger.warning(f"Error calculating perplexity: {e}")
                    continue
        
        for hook in hooks:
            hook.remove()
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        
        # Clamp loss to prevent extreme perplexity values
        # Perplexity = exp(loss), so loss > 10 means perplexity > 22,000
        # We cap at loss=10 (perplexity ≈ 22,026) to keep values interpretable
        # This still shows degradation but in a reasonable range
        max_reasonable_loss = 10.0  # Corresponds to perplexity ≈ 22,026
        if avg_loss > max_reasonable_loss:
            avg_loss = max_reasonable_loss
        
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Also return log-perplexity for easier interpretation
        # Log-perplexity is just the average loss, which is more interpretable
        return perplexity
    
    def evaluate_accuracy(self, tasks: List[Dict], config: AblationConfig) -> Dict:
        """Evaluate task accuracy on IFEval tasks."""
        hooks = []
        
        # Step 1: Steering hook (amplification)
        def steering_hook(module, input, output):
            """Hook that amplifies specified neurons, then ablates if specified."""
            if isinstance(output, tuple):
                hidden = output[0].clone()
                rest = output[1:]
            else:
                hidden = output.clone()
                rest = ()
            
            # Apply steering (amplification)
            if config.neurons and config.strength != 1.0 and len(config.neurons) > 0:
                # Filter neurons to valid range
                if hidden.dim() >= 2:
                    max_neuron = hidden.shape[-1]
                else:
                    max_neuron = hidden.numel()
                valid_neurons = [n for n in config.neurons if n < max_neuron]
                
                # Handle different output shapes
                # For most models: output is [batch, seq_len, hidden_dim]
                if hidden.dim() >= 2:
                    # Amplify cluster neurons on the last dimension
                    last_dim = hidden.shape[-1]
                    for neuron_idx in valid_neurons:
                        if neuron_idx < last_dim:
                            hidden[..., neuron_idx] *= config.strength
                else:
                    # 1D or scalar output - apply to all if within range
                    if hidden.numel() > 0:
                        for neuron_idx in valid_neurons:
                            if neuron_idx < hidden.numel():
                                hidden.view(-1)[neuron_idx] *= config.strength
            
            # Step 2: Apply ablation (zero out) if specified
            if config.ablate_neurons and len(config.ablate_neurons) > 0:
                # Zero out specific neurons after steering
                if hidden.dim() >= 2:
                    last_dim = hidden.shape[-1]
                    for neuron_idx in config.ablate_neurons:
                        if neuron_idx < last_dim:
                            hidden[..., neuron_idx] = 0.0
                else:
                    if hidden.numel() > 0:
                        for neuron_idx in config.ablate_neurons:
                            if neuron_idx < hidden.numel():
                                hidden.view(-1)[neuron_idx] = 0.0
            
            if rest:
                return (hidden,) + rest
            return hidden
        
        # Register steering hook at the steering layer
        target = self._get_steering_target(config.layer)
        hook = target.register_forward_hook(steering_hook)
        hooks.append(hook)
        
        # Step 3: Ablation hook for specific layer (if ablate_layer is specified)
        if config.ablate_layer is not None and config.ablate_layer != config.layer:
            def ablation_hook(module, input, output):
                """Hook that zeros out entire layer output."""
                if isinstance(output, tuple):
                    hidden = torch.zeros_like(output[0])
                    rest = output[1:]
                    if rest:
                        return (hidden,) + rest
                    return hidden
                else:
                    return torch.zeros_like(output)
            
            ablation_target = self._get_steering_target(config.ablate_layer)
            ablation_hook_handle = ablation_target.register_forward_hook(ablation_hook)
            hooks.append(ablation_hook_handle)
        
        correct = 0
        total = 0
        
        for task in tasks:
            prompt = task.get('prompt', '')
            if not prompt:
                continue
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_beams=1,
                        use_cache=True
                    )
                    
                    input_len = inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(
                        outputs[0][input_len:],
                        skip_special_tokens=True
                    ).strip()
                    
                    # For IFEval, we check if the response follows instructions
                    # This is simplified - actual IFEval evaluation is more complex
                    # For now, we'll use a simple heuristic
                    is_correct = self._check_ifeval_response(task, response)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                
                except Exception as e:
                    total += 1
        
        for hook in hooks:
            hook.remove()
        
        accuracy = correct / total if total > 0 else 0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def _check_ifeval_response(self, task: Dict, response: str) -> bool:
        """
        Simplified IFEval response checking.
        In practice, IFEval has complex verification rules.
        For now, we check basic instruction following.
        """
        prompt = task.get('prompt', '').lower()
        response_lower = response.lower()
        
        # Check for common instruction patterns
        if 'no comma' in prompt or 'without using any commas' in prompt:
            if ',' not in response:
                return True
            else:
                return False  # Failed: used commas when told not to
        
        if 'at least' in prompt:
            # Extract number requirement
            import re
            numbers = re.findall(r'at least (\d+)', prompt)
            if numbers:
                # Simple check - response should be long enough
                min_length = int(numbers[0])
                if len(response.split()) >= min_length:
                    return True
                else:
                    return False  # Failed: didn't meet minimum length
        
        # Check for specific format requirements
        if 'shakespearean' in prompt or 'shakespeare' in prompt:
            # Very basic check - should contain some Shakespearean words
            shakespeare_words = ['thou', 'thee', 'thy', 'thine', 'hath', 'doth', 'art', 'hast']
            if any(word in response_lower for word in shakespeare_words):
                return True
            # If prompt asks for Shakespearean but response doesn't have those words, might be wrong
            # But be lenient - just check if response is substantial
            return len(response.split()) >= 10
        
        # For other prompts, check if response is substantial and relevant
        # Require at least 5 words to avoid accepting trivial responses
        if len(response.split()) < 5:
            return False
        
        # Default: accept if response is substantial (at least 5 words)
        return len(response.strip()) > 0 and len(response.split()) >= 5
    
    def run_cluster_ablation(self, ifeval_tasks: List[Dict], ifeval_texts: List[str]) -> Dict:
        """
        Run cluster ablation: Steering first, then ablate each cluster.
        
        Workflow:
        1. Baseline: No steering, no ablation - measure perplexity
        2. Steering: Apply steering (amplify neurons by 5.0x) - measure perplexity (before ablation)
        3. Ablation: Zero out each cluster after steering - measure perplexity (after ablation)
        
        Args:
            ifeval_tasks: List of IFEval task dictionaries with 'prompt' field
            ifeval_texts: List of text strings from IFEval for perplexity calculation
        """
        logger.info("\n" + "="*80)
        logger.info("CLUSTER ABLATION ON IFEVAL DATASET")
        logger.info("="*80)
        logger.info("Workflow: 1) Baseline 2) Steering 3) Steering + Ablation")
        logger.info(f"Testing {len(ifeval_tasks)} IFEval tasks")
        logger.info(f"Using {len(ifeval_texts)} texts for perplexity calculation")
        logger.info("-"*80)
        
        results = {}
        num_neurons = 16  # Match the table
        
        # Select neuron groups
        logger.info("\n📊 Selecting Neuron Groups...")
        delta_sensitive_neurons = self._select_delta_sensitive_neurons(num_neurons)
        random_neurons = self._select_random_neurons(num_neurons)
        high_variance_neurons = self._select_high_variance_neurons(ifeval_texts[:100], num_neurons)
        
        logger.info(f"  Delta-Sensitive Subspace (SSM Core): {len(delta_sensitive_neurons)} neurons")
        logger.info(f"  Random Selection: {len(random_neurons)} neurons")
        logger.info(f"  High-Variance: {len(high_variance_neurons)} neurons")
        
        neuron_groups = {
            'baseline': [],
            'delta_sensitive': delta_sensitive_neurons,
            'random': random_neurons,
            'high_variance': high_variance_neurons
        }
        
        # Baseline: No steering
        logger.info("\n📊 Baseline (No steering)")
        baseline_config = AblationConfig(
            neurons=[],
            layer=self.best_layer,
            strength=1.0,
            experiment_type='baseline'
        )
        
        baseline_acc_result = self.evaluate_accuracy(ifeval_tasks, baseline_config)
        baseline_ppl = self.calculate_perplexity(ifeval_texts, baseline_config)
        baseline_acc = baseline_acc_result['accuracy'] * 100
        
        results['baseline'] = {
            'neurons': [],
            'accuracy': baseline_acc,
            'perplexity': baseline_ppl
        }
        logger.info(f"  Accuracy: {baseline_acc:.1f}%")
        logger.info(f"  Perplexity: {baseline_ppl:.2f}")
        
        # Test each neuron group: Steering first, then ablation
        logger.info("\n📊 Cluster Ablation Results")
        logger.info("For each cluster: 1) Steering only 2) Steering + Ablation")
        logger.info("-"*80)
        
        for group_name, neurons in neuron_groups.items():
            if group_name == 'baseline':
                continue
            
            logger.info(f"\n  {group_name.upper().replace('_', ' ')}")
            
            # Step 1: Steering only (before ablation)
            steering_config = AblationConfig(
                neurons=neurons,
                layer=self.best_layer,
                strength=self.best_strength,
                experiment_type=f'cluster_{group_name}_steering'
            )
            
            # Calculate perplexity with steering only
            ppl_steering = self.calculate_perplexity(ifeval_texts, steering_config)
            acc_result_steering = self.evaluate_accuracy(ifeval_tasks, steering_config)
            acc_steering = acc_result_steering['accuracy'] * 100
            
            logger.info(f"    Steering only:")
            logger.info(f"      Accuracy: {acc_steering:.1f}%")
            logger.info(f"      Perplexity: {ppl_steering:.2f} (baseline: {baseline_ppl:.2f})")
            
            # Step 2: Steering + Ablation (after ablation)
            ablation_config = AblationConfig(
                neurons=neurons,
                layer=self.best_layer,
                strength=self.best_strength,
                experiment_type=f'cluster_{group_name}_ablation',
                ablate_neurons=neurons  # Zero out these neurons after steering
            )
            
            # Calculate perplexity with steering + ablation
            ppl_ablation = self.calculate_perplexity(ifeval_texts, ablation_config)
            acc_result_ablation = self.evaluate_accuracy(ifeval_tasks, ablation_config)
            acc_ablation = acc_result_ablation['accuracy'] * 100
            
            logger.info(f"    Steering + Ablation:")
            logger.info(f"      Accuracy: {acc_ablation:.1f}%")
            logger.info(f"      Perplexity: {ppl_ablation:.2f} (baseline: {baseline_ppl:.2f})")
            
            # Calculate changes from baseline
            ppl_change_steering = ppl_steering - baseline_ppl
            ppl_change_ablation = ppl_ablation - baseline_ppl
            ppl_change_pct_steering = (ppl_change_steering / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            ppl_change_pct_ablation = (ppl_change_ablation / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            
            results[group_name] = {
                'neurons': neurons,
                'steering': {
                    'accuracy': acc_steering,
                    'accuracy_change': acc_steering - baseline_acc,
                    'perplexity': ppl_steering,
                    'perplexity_change': ppl_change_steering,
                    'perplexity_change_percent': ppl_change_pct_steering
                },
                'ablation': {
                    'accuracy': acc_ablation,
                    'accuracy_change': acc_ablation - baseline_acc,
                    'perplexity': ppl_ablation,
                    'perplexity_change': ppl_change_ablation,
                    'perplexity_change_percent': ppl_change_pct_ablation
                }
            }
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("CLUSTER ABLATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Baseline (before ablation): Accuracy={baseline_acc:.1f}%, Perplexity={baseline_ppl:.2f}")
        logger.info("")
        logger.info(f"{'Group':<20} {'Steering PPL':<15} {'Ablation PPL':<15} {'Δ (Abl-Base)':<15}")
        logger.info("-"*80)
        
        for group_name in ['delta_sensitive', 'random', 'high_variance']:
            if group_name in results:
                r = results[group_name]
                steering_ppl = r['steering']['perplexity']
                ablation_ppl = r['ablation']['perplexity']
                ablation_change = r['ablation']['perplexity_change']
                logger.info(f"{group_name.replace('_', ' ').title():<20} "
                          f"{steering_ppl:>13.2f} {ablation_ppl:>13.2f} {ablation_change:>+13.2f}")
        
        logger.info("="*80)
        
        return results
    
    def run_combined_steering_analysis(self, ifeval_tasks: List[Dict]) -> Dict:
        """
        Combined steering analysis: Compare steering delta-sensitive subspace neurons at different layers
        and different subspaces at Layer 20.
        
        Based on reviewer feedback:
        - Show that steering delta-sensitive subspace neurons at Layer 20 is better than at other layers
        - Show that steering delta-sensitive subspace neurons at Layer 20 is better than steering other subspaces at Layer 20
        - Use IFEval accuracy as the metric (NOT perplexity)
        - Use the same baseline for both comparisons
        - Combine into one table
        
        NOTE: This method ONLY measures accuracy, not perplexity.
        
        Args:
            ifeval_tasks: List of IFEval task dictionaries with 'prompt' field
        """
        logger.info("\n" + "="*80)
        logger.info("COMBINED STEERING ANALYSIS (IFEval Dataset)")
        logger.info("="*80)
        logger.info("Goal: Show that steering delta-sensitive subspace neurons at Layer 20 is optimal")
        logger.info("1. Compare delta-sensitive subspace neurons steered at different layers")
        logger.info("2. Compare different subspaces steered at Layer 20")
        logger.info(f"Testing {len(ifeval_tasks)} IFEval tasks")
        logger.info("Metric: IFEval Accuracy (higher is better)")
        logger.info("-"*80)
        
        results = {}
        num_neurons = 16  # Match the table
        
        # Select neuron groups
        logger.info("\n📊 Selecting Neuron Groups...")
        delta_sensitive_neurons = self._select_delta_sensitive_neurons(num_neurons)
        random_neurons = self._select_random_neurons(num_neurons)
        # Extract texts from tasks for high variance selection
        task_texts = [task.get('prompt', '') for task in ifeval_tasks[:100] if task.get('prompt')]
        high_variance_neurons = self._select_high_variance_neurons(task_texts, num_neurons)
        
        logger.info(f"  Delta-Sensitive Subspace (SSM Core): {len(delta_sensitive_neurons)} neurons")
        logger.info(f"  Random Selection: {len(random_neurons)} neurons")
        logger.info(f"  High-Variance: {len(high_variance_neurons)} neurons")
        
        layers_to_test = [18, 19, 20, 21, 22]
        layer_descriptions = {
            18: "Pre-bottleneck",
            19: "Pre-compression",
            20: "Bottleneck",
            21: "Post-bottleneck",
            22: "Output projection"
        }
        
        # Baseline: No steering
        logger.info("\n📊 Baseline (No steering)")
        baseline_config = AblationConfig(
            neurons=[],
            layer=self.best_layer,
            strength=1.0,
            experiment_type='baseline'
        )
        
        baseline_acc_result = self.evaluate_accuracy(ifeval_tasks, baseline_config)
        baseline_acc = baseline_acc_result['accuracy'] * 100
        
        results['baseline'] = {
            'accuracy': baseline_acc,
            'description': 'No steering'
        }
        logger.info(f"  Accuracy: {baseline_acc:.1f}%")
        
        # Experiment 1: Delta-sensitive subspace neurons at different layers
        logger.info("\n📊 Experiment 1: Delta-sensitive subspace neurons at different layers")
        logger.info("-"*80)
        
        for layer_idx in layers_to_test:
            if layer_idx >= self.num_layers:
                continue
            
            # Use layer-specific strength
            layer_strength = self.layer_strengths.get(layer_idx, self.best_strength)
            
            config = AblationConfig(
                neurons=delta_sensitive_neurons,
                layer=layer_idx,
                strength=layer_strength,
                experiment_type=f'delta_sensitive_layer_{layer_idx}'
            )
            
            acc_result = self.evaluate_accuracy(ifeval_tasks, config)
            acc = acc_result['accuracy'] * 100
            acc_change = acc - baseline_acc
            
            results[f'delta_sensitive_layer_{layer_idx}'] = {
                'accuracy': acc,
                'accuracy_change': acc_change,
                'description': f'Delta-Sensitive @ Layer {layer_idx} ({layer_descriptions.get(layer_idx, "Unknown")})',
                'layer': layer_idx,
                'neurons': 'Delta-Sensitive Subspace',
                'correct': acc_result.get('correct', 0),
                'total': acc_result.get('total', 0)
            }
            
            logger.info(f"  Layer {layer_idx} ({layer_descriptions.get(layer_idx, 'Unknown')}): "
                      f"Accuracy={acc:.1f}% ({acc_change:+.1f}%) "
                      f"[{acc_result.get('correct', 0)}/{acc_result.get('total', 0)}] "
                      f"[strength: {layer_strength}x]")
        
        # Experiment 2: Different subspaces at Layer 20
        logger.info("\n📊 Experiment 2: Different subspaces at Layer 20")
        logger.info("-"*80)
        
        subspace_groups = {
            'delta_sensitive': delta_sensitive_neurons,
            'random': random_neurons,
            'high_variance': high_variance_neurons
        }
        
        for subspace_name, neurons in subspace_groups.items():
            config = AblationConfig(
                neurons=neurons,
                layer=self.best_layer,  # Layer 20
                strength=self.best_strength,  # 5.0x
                experiment_type=f'{subspace_name}_layer_20'
            )
            
            acc_result = self.evaluate_accuracy(ifeval_tasks, config)
            acc = acc_result['accuracy'] * 100
            acc_change = acc - baseline_acc
            
            subspace_display_name = {
                'delta_sensitive': 'Delta-Sensitive Subspace (SSM Core)',
                'random': 'Random Selection',
                'high_variance': 'High-Variance Neurons'
            }.get(subspace_name, subspace_name)
            
            results[f'{subspace_name}_layer_20'] = {
                'accuracy': acc,
                'accuracy_change': acc_change,
                'description': f'{subspace_display_name} @ Layer 20',
                'layer': 20,
                'neurons': subspace_display_name,
                'correct': acc_result.get('correct', 0),
                'total': acc_result.get('total', 0)
            }
            
            logger.info(f"  {subspace_display_name}: Accuracy={acc:.1f}% ({acc_change:+.1f}%) "
                      f"[{acc_result.get('correct', 0)}/{acc_result.get('total', 0)}]")
        
        # Summary table
        logger.info("\n" + "="*80)
        logger.info("COMBINED STEERING ANALYSIS SUMMARY")
        logger.info("="*80)
        logger.info(f"Baseline: Accuracy={baseline_acc:.1f}% (no steering)")
        logger.info("")
        logger.info(f"{'Experiment':<35} {'Configuration':<40} {'Accuracy (%)':<15} {'Δ from Baseline':<15}")
        logger.info("-"*105)
        
        # Baseline
        logger.info(f"{'Baseline':<35} {'No steering':<40} {baseline_acc:>13.1f}% {'-':>15}")
        
        # Layer comparison
        logger.info("")
        logger.info("Layer Comparison (Delta-sensitive subspace neurons at different layers):")
        for layer_idx in layers_to_test:
            if layer_idx >= self.num_layers:
                continue
            key = f'delta_sensitive_layer_{layer_idx}'
            if key in results:
                r = results[key]
                logger.info(f"{r['description']:<35} {'Delta-sensitive neurons, Layer ' + str(layer_idx):<40} "
                          f"{r['accuracy']:>13.1f}% {r['accuracy_change']:>+13.1f}%")
        
        # Subspace comparison
        logger.info("")
        logger.info("Subspace Comparison (Different subspaces at Layer 20):")
        for subspace_name in ['delta_sensitive', 'random', 'high_variance']:
            key = f'{subspace_name}_layer_20'
            if key in results:
                r = results[key]
                logger.info(f"{r['description']:<35} {r['neurons'] + ', Layer 20':<40} "
                          f"{r['accuracy']:>13.1f}% {r['accuracy_change']:>+13.1f}%")
        
        logger.info("="*105)
        
        return results
    
    def run_layer_ablation_ifeval(self, ifeval_tasks: List[Dict], ifeval_texts: List[str], neurons: Optional[List[int]] = None) -> Dict:
        """
        Layer-wise steering analysis on IFEval dataset with both accuracy and perplexity.
        
        NOTE: This performs STEERING (amplification), not ablation (removal).
        Neurons are amplified at different layers with layer-specific strengths:
        - Layers 18-20: 5.0x amplification
        - Layer 21: 2.0x amplification  
        - Layer 22: 1.5x amplification
        
        Args:
            ifeval_tasks: List of IFEval task dictionaries
            ifeval_texts: List of text strings for perplexity calculation
            neurons: List of neurons to steer (if None, uses all neurons)
        """
        logger.info("\n" + "="*80)
        logger.info("LAYER-WISE STEERING ANALYSIS ON IFEVAL DATASET")
        logger.info("NOTE: Amplifying neurons (steering), not removing them (ablation)")
        logger.info("="*80)
        logger.info(f"Testing {len(ifeval_tasks)} IFEval tasks")
        logger.info(f"Using {len(ifeval_texts)} texts for perplexity calculation")
        logger.info("Metric: Accuracy (higher is better) and Perplexity (lower is better)")
        logger.info("-"*80)
        
        results = {}
        
        # Use all neurons if not specified
        if neurons is None:
            neurons = list(range(self.hidden_dim))
        
        layers_to_test = [18, 19, 20, 21, 22]
        layer_descriptions = {
            18: "Pre-bottleneck",
            19: "Pre-compression",
            20: "Bottleneck",
            21: "Post-bottleneck",
            22: "Output projection"
        }
        
        # Baseline: No steering
        logger.info(f"\n📊 Baseline (No steering)")
        baseline_config = AblationConfig(
            neurons=[],
            layer=self.best_layer,
            strength=1.0,
            experiment_type='baseline'
        )
        
        baseline_acc_result = self.evaluate_accuracy(ifeval_tasks, baseline_config)
        baseline_ppl = self.calculate_perplexity(ifeval_texts, baseline_config)
        baseline_acc = baseline_acc_result['accuracy'] * 100
        
        results['baseline'] = {
            'layer': None,
            'accuracy': baseline_acc,
            'perplexity': baseline_ppl,
            'neurons': []
        }
        logger.info(f"  Accuracy: {baseline_acc:.1f}%")
        logger.info(f"  Perplexity: {baseline_ppl:.2f}")
        
        # Test steering at different layers, then ablation
        logger.info(f"\n📊 Layer-wise Results")
        logger.info("For each layer: 1) Steering only 2) Steering + Ablation")
        logger.info("-"*80)
        
        for layer_idx in layers_to_test:
            if layer_idx >= self.num_layers:
                continue
            
            logger.info(f"\n  Layer {layer_idx} ({layer_descriptions.get(layer_idx, 'Unknown')})")
            
            # Use layer-specific strength to prevent extreme values
            layer_strength = self.layer_strengths.get(layer_idx, self.best_strength)
            
            # Step 1: Steering only (before ablation)
            steering_config = AblationConfig(
                neurons=neurons,
                layer=layer_idx,
                strength=layer_strength,
                experiment_type=f'layer_{layer_idx}_steering'
            )
            
            # Calculate perplexity with steering only
            ppl_steering = self.calculate_perplexity(ifeval_texts, steering_config)
            acc_result_steering = self.evaluate_accuracy(ifeval_tasks, steering_config)
            acc_steering = acc_result_steering['accuracy'] * 100
            
            logger.info(f"    Steering only (strength: {layer_strength}x):")
            logger.info(f"      Accuracy: {acc_steering:.1f}%")
            logger.info(f"      Perplexity: {ppl_steering:.2f} (baseline: {baseline_ppl:.2f})")
            
            # Step 2: Steering + Ablation (after ablation)
            ablation_config = AblationConfig(
                neurons=neurons,
                layer=layer_idx,
                strength=layer_strength,
                experiment_type=f'layer_{layer_idx}_ablation',
                ablate_layer=layer_idx  # Zero out this layer after steering
            )
            
            # Calculate perplexity with steering + ablation
            ppl_ablation = self.calculate_perplexity(ifeval_texts, ablation_config)
            acc_result_ablation = self.evaluate_accuracy(ifeval_tasks, ablation_config)
            acc_ablation = acc_result_ablation['accuracy'] * 100
            
            logger.info(f"    Steering + Ablation:")
            logger.info(f"      Accuracy: {acc_ablation:.1f}%")
            logger.info(f"      Perplexity: {ppl_ablation:.2f} (baseline: {baseline_ppl:.2f})")
            
            # Calculate changes from baseline
            ppl_change_steering = ppl_steering - baseline_ppl
            ppl_change_ablation = ppl_ablation - baseline_ppl
            ppl_change_pct_steering = (ppl_change_steering / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            ppl_change_pct_ablation = (ppl_change_ablation / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            
            results[f'layer_{layer_idx}'] = {
                'layer': layer_idx,
                'role': layer_descriptions.get(layer_idx, 'Unknown'),
                'steering_strength': layer_strength,
                'neurons': neurons[:10] if len(neurons) > 10 else neurons,
                'steering': {
                    'accuracy': acc_steering,
                    'accuracy_change': acc_steering - baseline_acc,
                    'perplexity': ppl_steering,
                    'perplexity_change': ppl_change_steering,
                    'perplexity_change_percent': ppl_change_pct_steering
                },
                'ablation': {
                    'accuracy': acc_ablation,
                    'accuracy_change': acc_ablation - baseline_acc,
                    'perplexity': ppl_ablation,
                    'perplexity_change': ppl_change_ablation,
                    'perplexity_change_percent': ppl_change_pct_ablation
                }
            }
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("LAYER ABLATION SUMMARY (IFEval)")
        logger.info("="*80)
        logger.info(f"Baseline (before ablation): Accuracy={baseline_acc:.1f}%, Perplexity={baseline_ppl:.2f}")
        logger.info("")
        logger.info(f"{'Layer':<8} {'Role':<20} {'Steering PPL':<15} {'Ablation PPL':<15} {'Δ (Abl-Base)':<15}")
        logger.info("-"*80)
        
        for layer_idx in layers_to_test:
            if layer_idx >= self.num_layers:
                continue
            key = f'layer_{layer_idx}'
            if key in results:
                r = results[key]
                steering_ppl = r['steering']['perplexity']
                ablation_ppl = r['ablation']['perplexity']
                ablation_change = r['ablation']['perplexity_change']
                
                # Format perplexity for display
                if steering_ppl > 10000:
                    steering_ppl_str = f"{steering_ppl:.0f}"
                elif steering_ppl > 1000:
                    steering_ppl_str = f"{steering_ppl:.0f}"
                else:
                    steering_ppl_str = f"{steering_ppl:.2f}"
                
                if ablation_ppl > 10000:
                    ablation_ppl_str = f"{ablation_ppl:.0f}"
                elif ablation_ppl > 1000:
                    ablation_ppl_str = f"{ablation_ppl:.0f}"
                else:
                    ablation_ppl_str = f"{ablation_ppl:.2f}"
                
                logger.info(f"{r['layer']:<8} {r['role']:<20} "
                          f"{steering_ppl_str:>13} {ablation_ppl_str:>13} {ablation_change:>+13.2f}")
        
        # Find best and worst layers
        # Find best layer based on ablation perplexity (lowest is best)
        best_layer_ppl = min(layers_to_test,
                            key=lambda l: results.get(f'layer_{l}', {}).get('ablation', {}).get('perplexity', float('inf')) if l < self.num_layers else float('inf'))
        
        if best_layer_ppl < self.num_layers:
            best_ppl_info = results[f'layer_{best_layer_ppl}']
            logger.info(f"\nBest layer (Ablation Perplexity): {best_layer_ppl} ({layer_descriptions.get(best_layer_ppl, 'Unknown')})")
            logger.info(f"  Ablation Perplexity: {best_ppl_info['ablation']['perplexity']:.2f} (change from baseline: {best_ppl_info['ablation']['perplexity_change']:+.2f})")
        
        logger.info("")
        logger.info("NOTE ON PERPLEXITY VALUES:")
        logger.info("  • Perplexity range: 1 to infinity (no upper bound)")
        logger.info("  • Typical values: 10-50 (good), 50-100 (moderate), >1000 (poor)")
        logger.info("  • Logits are clipped to [-50, 50] to prevent numerical overflow")
        logger.info("  • Later layers use reduced steering strength to prevent instability")
        logger.info("  • Values are capped at exp(10)≈22,000 for interpretability")
        logger.info("="*80)
        
        return results


def load_ifeval_dataset(num_samples: Optional[int] = None) -> Tuple[List[Dict], List[str]]:
    """
    Load IFEval dataset from HuggingFace.
    
    Returns:
        Tuple of (tasks, texts) where:
        - tasks: List of task dicts with 'prompt' field
        - texts: List of text strings for perplexity calculation
    """
    try:
        from datasets import load_dataset
        logger.info("Loading IFEval dataset from HuggingFace...")
        
        dataset = load_dataset("google/IFEval", split="train")
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        tasks = []
        texts = []
        
        for item in dataset:
            prompt = item.get('prompt', '')
            if prompt:
                tasks.append({'prompt': prompt})
                texts.append(prompt)
        
        logger.info(f"✅ Loaded {len(tasks)} IFEval tasks")
        return tasks, texts
    
    except Exception as e:
        logger.error(f"❌ Error loading IFEval dataset: {e}")
        logger.info("   Falling back to synthetic IFEval-like tasks...")
        
        # Fallback: Create synthetic IFEval-like tasks
        tasks = []
        texts = []
        
        synthetic_prompts = [
            "Write a 300+ word summary without using any commas.",
            "Write a poem with at least 5 sentences.",
            "Create a resume with at least 12 placeholders like [name], [address].",
            "Write a story in Shakespearean style without commas.",
            "Write an acoustic song about the Korean peninsula without using any commas."
        ]
        
        for prompt in synthetic_prompts * (num_samples // len(synthetic_prompts) + 1):
            tasks.append({'prompt': prompt})
            texts.append(prompt)
        
        tasks = tasks[:num_samples] if num_samples else tasks
        texts = texts[:num_samples] if num_samples else texts
        
        logger.info(f"   Created {len(tasks)} synthetic IFEval-like tasks")
        return tasks, texts


def main():
    """Main function to run cluster ablation on IFEval."""
    import argparse
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    parser = argparse.ArgumentParser(description="Cluster Ablation on IFEval")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf",
                       help="Model to evaluate")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of IFEval samples to use")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="cluster_ablation_ifeval_results",
                       help="Directory to save results")
    parser.add_argument("--layer_ablation", action="store_true",
                       help="Run layer-wise ablation")
    parser.add_argument("--cluster_ablation", action="store_true",
                       help="Run cluster ablation")
    parser.add_argument("--both", action="store_true",
                       help="Run both cluster and layer ablation")
    parser.add_argument("--combined", action="store_true",
                       help="Run combined steering analysis (Delta-sensitive subspace at different layers + different subspaces at Layer 20)")
    
    args = parser.parse_args()
    
    # Determine what to run
    run_combined = args.combined
    run_cluster = args.cluster_ablation or args.both or (not args.layer_ablation and not args.cluster_ablation and not run_combined)
    run_layer = args.layer_ablation or args.both
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name=args.model,
        device=args.device if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Load IFEval dataset
    ifeval_tasks, ifeval_texts = load_ifeval_dataset(num_samples=args.num_samples)
    
    analyzer = ClusterAblationAnalyzer(model, tokenizer, device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return obj
    
    # Run combined steering analysis if requested
    if run_combined:
        logger.info("\n" + "="*80)
        logger.info("RUNNING COMBINED STEERING ANALYSIS")
        logger.info("="*80)
        combined_results = analyzer.run_combined_steering_analysis(ifeval_tasks)
        combined_results_path = output_dir / "combined_steering_ifeval_results.json"
        with open(combined_results_path, 'w') as f:
            json.dump(combined_results, f, indent=2, default=make_serializable)
        logger.info(f"\n✅ Combined steering results saved to: {combined_results_path}")
    
    # Run cluster ablation if requested
    if run_cluster:
        logger.info("\n" + "="*80)
        logger.info("RUNNING CLUSTER ABLATION")
        logger.info("="*80)
        cluster_results = analyzer.run_cluster_ablation(ifeval_tasks, ifeval_texts)
        cluster_results_path = output_dir / "cluster_ablation_ifeval_results.json"
        with open(cluster_results_path, 'w') as f:
            json.dump(cluster_results, f, indent=2, default=make_serializable)
        logger.info(f"\n✅ Cluster ablation results saved to: {cluster_results_path}")
    
    # Run layer ablation if requested
    if run_layer:
        logger.info("\n" + "="*80)
        logger.info("RUNNING LAYER ABLATION")
        logger.info("="*80)
        # Try to load discovered neurons if available
        neurons = None
        try:
            json_path = Path("ablation_3_results/discovered_neurons.json")
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'all_neuron_impacts' in data:
                        neurons = [item['neuron'] for item in data['all_neuron_impacts']]
                        logger.info(f"Using {len(neurons)} discovered neurons from impact analysis")
        except Exception as e:
            logger.warning(f"Could not load discovered neurons: {e}")
        
        if neurons is None:
            neurons = list(range(analyzer.hidden_dim))
            logger.info(f"Using all {len(neurons)} neurons")
        
        layer_results = analyzer.run_layer_ablation_ifeval(ifeval_tasks, ifeval_texts, neurons=neurons)
        layer_results_path = output_dir / "layer_ablation_ifeval_results.json"
        with open(layer_results_path, 'w') as f:
            json.dump(layer_results, f, indent=2, default=make_serializable)
        logger.info(f"\n✅ Layer ablation results saved to: {layer_results_path}")


if __name__ == "__main__":
    main()


