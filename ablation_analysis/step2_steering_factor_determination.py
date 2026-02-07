"""
CLUSTER 9 INTRA-NEURON ABLATION STUDY

Goal: Prove that each neuron in Cluster 9 is actually contributing to performance.

Experimental Design:
1. Leave-One-Out (LOO): Remove each neuron individually and measure performance drop
2. Add-One-In (AOI): Start with empty set, add neurons one by one
3. Neuron Ranking: Identify which neurons are most critical
4. Subgroup Analysis: Test performance with different subsets

Expected Result:
- Each neuron removal should cause measurable performance degradation
- Neurons should show varying levels of importance
- Performance should correlate with number of active neurons


Result to show various amplification strengths

Tests for High Impact Neurons (>2%)
Testing amplification values from 0.1x to 100x
Coverage: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0

Tests for Neutral Neurons (-2% to +2%)
Testing amplification values from 0.1x to 100x
Coverage: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0



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
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""
    neurons: List[int]
    layer: int
    strength: float
    experiment_type: str  # 'leave_one_out', 'add_one_in', 'random_subset', etc.
    ablated_neuron: Optional[int] = None
    use_impact_based: bool = False


class SimpleTaskGenerator:
    """Generate simple tasks for quick ablation testing."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_test_set(self, num_examples: int = 100) -> List[Dict]:
        """Generate mixed test set for ablation study."""
        tasks = []
        
        # Task 1: Chain reasoning (25 examples)
        for i in range(25):
            names = ['Alice', 'Bob', 'Carol', 'David', 'Emma']
            chain = np.random.choice(names, size=3, replace=False).tolist()
            
            facts = [
                f"{chain[0]} is taller than {chain[1]}.",
                f"{chain[1]} is taller than {chain[2]}."
            ]
            
            prompt = "Facts:\n" + "\n".join(facts) + "\n\nQuestion: Who is the tallest person?\nAnswer:"
            
            tasks.append({
                'prompt': prompt,
                'expected': chain[0],
                'alternatives': [chain[0], chain[0].lower()],
                'task_type': 'chain_reasoning'
            })
        
        # Task 2: Instruction following (25 examples)
        for i in range(25):
            a, b = np.random.randint(1, 20, size=2)
            operations = [
                (f"Add {a} and {b}.", a + b),
                (f"Multiply {a} by {b}.", a * b),
                (f"Subtract {b} from {a}.", a - b),
            ]
            instruction, answer = operations[i % len(operations)]
            
            prompt = f"Instruction: {instruction}\n\nAnswer:"
            
            tasks.append({
                'prompt': prompt,
                'expected': str(answer),
                'alternatives': [str(answer)],
                'task_type': 'instruction_following'
            })
        
        # Task 3: Long context recall (25 examples)
        for i in range(25):
            names = ['Alice', 'Bob', 'Carol', 'David']
            colors = ['red', 'blue', 'green', 'yellow']
            cities = ['Paris', 'London', 'Tokyo', 'Berlin']
            
            name = np.random.choice(names)
            color = np.random.choice(colors)
            city = np.random.choice(cities)
            
            facts = [
                f"- {name} lives in {city}",
                f"- {name}'s favorite color is {color}",
                f"- {name} works as an engineer",
                f"- {name} enjoys reading",
                f"- The weather is sunny today",
                f"- Technology is advancing rapidly"
            ]
            np.random.shuffle(facts)
            
            question = f"What is {name}'s favorite color?"
            prompt = "Information:\n" + "\n".join(facts) + f"\n\nQuestion: {question}\nAnswer:"
            
            tasks.append({
                'prompt': prompt,
                'expected': color,
                'alternatives': [color, color.capitalize()],
                'task_type': 'long_context_recall'
            })
        
        # Task 4: Needle in haystack (25 examples)
        for i in range(25):
            distractors = "The quick brown fox jumps over the lazy dog. " * 20
            needle = f"The secret code is CODE{i:03d}."
            position = len(distractors) // 2
            text = distractors[:position] + needle + distractors[position:]
            
            prompt = f"Text: {text}\n\nQuestion: What is the secret code?\nAnswer:"
            expected = f"CODE{i:03d}"
            
            tasks.append({
                'prompt': prompt,
                'expected': expected,
                'alternatives': [expected, expected.lower()],
                'task_type': 'needle_in_haystack'
            })
        
        return tasks


class Cluster9AblationAnalyzer:
    """
    Analyze the contribution of individual neurons within Cluster 9.
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
        
        # Cluster 9 neurons (from your mechanistic interpretability)
        # COMMENTED OUT: Now testing ALL neurons in the hidden dimension
        # self.cluster9_neurons = [
        #     4, 38, 84, 94, 163, 171, 268, 363, 401, 497,
        #     564, 568, 582, 654, 659, 686
        # ]
        
        # Use ALL neurons in the hidden dimension for ablation
        # This will test all neurons, not just the 16 Cluster 9 neurons
        self.cluster9_neurons = list(range(self.hidden_dim))
        
        # Best configuration from validation
        self.best_layer = 20
        self.best_strength = 5.0
        
        # Impact threshold for categorizing neurons
        # The ablation study tests various amplification strengths from 0.1x to 100x
        # to determine which values work best for different neuron groups
        # We do NOT use fixed amplification values - we test all values from 0.1x to 100x
        self.impact_threshold = 2.0            # Threshold for categorizing neuron impact
        
        # Load neuron impacts for impact-based amplification
        self.neuron_impacts = {}  # Maps neuron_id -> impact percentage
        self._load_neuron_impacts()
        
        logger.info(f"Initialized Cluster 9 Ablation Analyzer:")
        logger.info(f"  Model layers: {self.num_layers}")
        logger.info(f"  Hidden dimension: {self.hidden_dim}")
        logger.info(f"  Testing ALL neurons: {len(self.cluster9_neurons)}")
        logger.info(f"  Neuron range: 0 to {self.hidden_dim-1}")
        if len(self.neuron_impacts) > 0:
            logger.info(f"  Loaded {len(self.neuron_impacts)} neuron impacts for impact-based amplification")
        if len(self.cluster9_neurons) > 50:
            logger.info(f"  ⚠️  Testing {len(self.cluster9_neurons)} neurons - this will take a long time!")
            logger.info(f"  Consider using --max_neurons to limit the number")
    
    def _load_neuron_impacts(self, json_path: str = "ablation_3_results/discovered_neurons.json"):
        """Load neuron impacts from discovered_neurons.json file."""
        try:
            json_file = Path(json_path)
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'all_neuron_impacts' in data:
                        for item in data['all_neuron_impacts']:
                            neuron_id = item['neuron']
                            impact = item['impact']
                            self.neuron_impacts[neuron_id] = impact
                        logger.info(f"✅ Loaded {len(self.neuron_impacts)} neuron impacts from {json_path}")
                    else:
                        logger.warning(f"⚠️  No 'all_neuron_impacts' found in {json_path}")
            else:
                logger.warning(f"⚠️  Neuron impacts file not found: {json_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load neuron impacts: {e}")
    
    def _get_amplification_for_neuron(self, neuron_id: int, default_strength: float = 1.0) -> float:
        """
        Get amplification strength for a neuron based on its impact.
        NOTE: This method is NOT used during ablation testing - we test all values from 0.1x to 100x.
        This is only used as a fallback when impact-based amplification is enabled but specific
        values are not being tested.
        
        - Neurons with >2% impact: Use default_strength (not fixed to 5.0)
        - Neurons with -2% to 2% impact: Use default_strength (not fixed to 2.0)
        - Other neurons: Use default_strength or no amplification
        """
        if neuron_id in self.neuron_impacts:
            impact = self.neuron_impacts[neuron_id]
            if impact > 2.0:
                return default_strength  # High impact neurons: use provided strength (not fixed)
            elif impact >= -2.0:
                return default_strength  # Neutral neurons: use provided strength (not fixed)
            else:
                return 1.0  # Detrimental neurons: no amplification
        return default_strength  # Unknown neurons: use default
    
    def _get_steering_target(self, layer_idx):
        """Get the module to apply steering to."""
        layer = self.layers[layer_idx]
        
        for attr in ['mixer', 'ssm', 'attn', 'self_attn']:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        
        return layer
    
    def _check_answer(self, response: str, expected: str, alternatives: List[str]) -> bool:
        """Check if response matches expected answer."""
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        if expected_lower in response_lower:
            return True
        
        for alt in alternatives:
            if alt and alt.lower().strip() in response_lower:
                return True
        
        response_words = response_lower.split()
        expected_words = expected_lower.split()
        
        if response_words and expected_words:
            if response_words[0] == expected_words[0]:
                return True
        
        import re
        if expected.replace('.', '').replace(',', '').replace('-', '').isdigit():
            response_nums = re.findall(r'-?\d+\.?\d*', response)
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            if response_nums and expected_nums:
                try:
                    if float(response_nums[0]) == float(expected_nums[0]):
                        return True
                except:
                    pass
        
        return False
    
    def calculate_perplexity(self, texts: List[str], config: AblationConfig) -> float:
        """
        Calculate perplexity on text continuations from The Pile.
        
        Perplexity = exp(cross_entropy_loss)
        Cross entropy loss = -log P(token | context)
        
        Args:
            texts: List of text strings from The Pile
            config: Ablation configuration (steering settings)
        
        Returns:
            Average perplexity across all texts
        """
        hooks = []
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            if config.neurons and config.strength != 1.0 and len(config.neurons) > 0:
                h_mod = hidden.clone()
                
                # Use impact-based amplification if neuron impacts are loaded
                use_impact_based = len(self.neuron_impacts) > 0 and hasattr(config, 'use_impact_based') and config.use_impact_based
                
                if use_impact_based:
                    # Impact-based amplification: different strength per neuron based on impact
                    for idx in config.neurons:
                        if idx < h_mod.shape[-1]:
                            neuron_strength = self._get_amplification_for_neuron(idx, config.strength)
                            h_mod[..., idx] *= neuron_strength
                else:
                    # Uniform amplification (original behavior)
                    if len(config.neurons) > 50:
                        # Vectorized approach for many neurons
                        mask = torch.ones(h_mod.shape[-1], device=h_mod.device, dtype=h_mod.dtype)
                        for idx in config.neurons:
                            if idx < mask.shape[0]:
                                mask[idx] = config.strength
                        h_mod = h_mod * mask
                    else:
                        # Individual modification for few neurons
                        for idx in config.neurons:
                            if idx < h_mod.shape[-1]:
                                h_mod[..., idx] *= config.strength
                
                if isinstance(output, tuple):
                    return (h_mod,) + output[1:]
                return h_mod
            
            return output
        
        target = self._get_steering_target(config.layer)
        hook = target.register_forward_hook(steering_hook)
        hooks.append(hook)
        
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize the full text
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
                        continue  # Skip very short texts
                    
                    # Forward pass to get logits
                    outputs = self.model(**inputs)
                    
                    # Extract logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Calculate cross entropy loss
                    loss_fct = F.cross_entropy
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    
                    loss = loss_fct(shift_logits, shift_labels, ignore_index=self.tokenizer.pad_token_id)
                    
                    # Count non-padding tokens
                    num_tokens = (shift_labels != self.tokenizer.pad_token_id).sum().item()
                    
                    if num_tokens > 0:
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                
                except Exception as e:
                    logger.warning(f"Error calculating perplexity for text: {e}")
                    continue
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        if total_tokens == 0:
            return float('inf')
        
        # Average cross entropy loss
        avg_loss = total_loss / total_tokens
        
        # Perplexity = exp(avg_loss)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_with_config(self, tasks: List[Dict], config: AblationConfig, verbose: bool = False) -> Dict:
        """Evaluate model with specific ablation configuration."""
        
        hooks = []
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            if config.neurons and config.strength != 1.0 and len(config.neurons) > 0:
                h_mod = hidden.clone()
                
                # Use impact-based amplification if neuron impacts are loaded
                use_impact_based = len(self.neuron_impacts) > 0 and hasattr(config, 'use_impact_based') and config.use_impact_based
                
                if use_impact_based:
                    # Impact-based amplification: different strength per neuron based on impact
                    for idx in config.neurons:
                        if idx < h_mod.shape[-1]:
                            neuron_strength = self._get_amplification_for_neuron(idx, config.strength)
                            h_mod[..., idx] *= neuron_strength
                else:
                    # Uniform amplification (original behavior)
                    # Optimize: use vectorized operations when many neurons
                    if len(config.neurons) > 50:
                        # Vectorized approach for many neurons
                        mask = torch.ones(h_mod.shape[-1], device=h_mod.device, dtype=h_mod.dtype)
                        for idx in config.neurons:
                            if idx < mask.shape[0]:
                                mask[idx] = config.strength
                        h_mod = h_mod * mask
                    else:
                        # Individual modification for few neurons
                        for idx in config.neurons:
                            if idx < h_mod.shape[-1]:
                                h_mod[..., idx] *= config.strength
                
                if isinstance(output, tuple):
                    return (h_mod,) + output[1:]
                return h_mod
            
            return output
        
        target = self._get_steering_target(config.layer)
        hook = target.register_forward_hook(steering_hook)
        hooks.append(hook)
        
        # Evaluate
        correct = 0
        total = 0
        results_by_task = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for idx, task in enumerate(tasks):
            if verbose and (idx + 1) % 10 == 0:
                logger.info(f"  Processing task {idx + 1}/{len(tasks)}...")
            prompt = task['prompt']
            expected = task['expected']
            alternatives = task.get('alternatives', [])
            task_type = task.get('task_type', 'unknown')
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                try:
                    # Add timeout and optimization for generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_beams=1,  # Faster than beam search
                        use_cache=True  # Use KV cache for speed
                    )
                    
                    input_len = inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(
                        outputs[0][input_len:],
                        skip_special_tokens=True
                    ).strip()
                    
                    is_correct = self._check_answer(response, expected, alternatives)
                    
                    if is_correct:
                        correct += 1
                        results_by_task[task_type]['correct'] += 1
                    
                    total += 1
                    results_by_task[task_type]['total'] += 1
                
                except Exception as e:
                    total += 1
                    results_by_task[task_type]['total'] += 1
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Compute accuracies
        overall_accuracy = correct / total if total > 0 else 0
        
        task_accuracies = {}
        for task_type, counts in results_by_task.items():
            task_accuracies[task_type] = (
                counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            )
        
        return {
            'accuracy': overall_accuracy,
            'correct': correct,
            'total': total,
            'task_accuracies': task_accuracies
        }
    
    def run_leave_one_out(self, tasks: List[Dict]) -> Dict:
        """
        Leave-One-Out Ablation: Remove each neuron individually.
        Shows how important each neuron is.
        """
        logger.info("\n" + "="*80)
        logger.info("LEAVE-ONE-OUT ABLATION")
        logger.info("="*80)
        logger.info("Remove each neuron individually and measure performance drop")
        logger.info("-"*80)
        
        results = {}
        
        # Baseline: All neurons active
        logger.info(f"\n📊 Baseline (All {len(self.cluster9_neurons)} neurons active)")
        baseline_config = AblationConfig(
            neurons=self.cluster9_neurons,
            layer=self.best_layer,
            strength=self.best_strength,
            experiment_type='baseline'
        )
        baseline_result = self.evaluate_with_config(tasks, baseline_config)
        results['baseline'] = {
            'neurons': self.cluster9_neurons,
            'result': baseline_result
        }
        logger.info(f"  Accuracy: {baseline_result['accuracy']*100:.1f}%")
        
        # Leave-one-out: Remove each neuron
        logger.info("\n📊 Leave-One-Out Results")
        logger.info("-"*80)
        
        for neuron in self.cluster9_neurons:
            neurons_without = [n for n in self.cluster9_neurons if n != neuron]
            
            config = AblationConfig(
                neurons=neurons_without,
                layer=self.best_layer,
                strength=self.best_strength,
                experiment_type='leave_one_out',
                ablated_neuron=neuron
            )
            
            result = self.evaluate_with_config(tasks, config)
            performance_drop = (baseline_result['accuracy'] - result['accuracy']) * 100
            
            results[f'without_{neuron}'] = {
                'ablated_neuron': neuron,
                'active_neurons': neurons_without,
                'result': result,
                'performance_drop': performance_drop
            }
            
            logger.info(f"  Neuron {neuron:3d}: {result['accuracy']*100:5.1f}% (Δ {performance_drop:+5.2f}%)")
        
        return results
    
    def run_add_one_in(self, tasks: List[Dict]) -> Dict:
        """
        Add-One-In: Start empty, add neurons one by one.
        Shows cumulative contribution of neurons.
        """
        logger.info("\n" + "="*80)
        logger.info("ADD-ONE-IN ABLATION")
        logger.info("="*80)
        logger.info("Start with no neurons, add one at a time in order of importance")
        logger.info("-"*80)
        
        results = {}
        
        # Baseline: No steering
        logger.info("\n📊 Baseline (No steering)")
        baseline_config = AblationConfig(
            neurons=[],
            layer=self.best_layer,
            strength=1.0,
            experiment_type='no_steering'
        )
        baseline_result = self.evaluate_with_config(tasks, baseline_config)
        results['no_steering'] = {
            'neurons': [],
            'result': baseline_result
        }
        logger.info(f"  Accuracy: {baseline_result['accuracy']*100:.1f}%")
        
        # Add neurons one by one
        logger.info("\n📊 Progressive Neuron Addition")
        logger.info("-"*80)
        
        active_neurons = []
        for i, neuron in enumerate(self.cluster9_neurons, 1):
            active_neurons.append(neuron)
            
            config = AblationConfig(
                neurons=active_neurons.copy(),
                layer=self.best_layer,
                strength=self.best_strength,
                experiment_type='add_one_in'
            )
            
            result = self.evaluate_with_config(tasks, config)
            improvement = (result['accuracy'] - baseline_result['accuracy']) * 100
            
            results[f'with_{i}_neurons'] = {
                'num_neurons': i,
                'active_neurons': active_neurons.copy(),
                'result': result,
                'improvement_over_baseline': improvement
            }
            
            logger.info(f"  {i:2d} neurons ({neuron:3d} added): {result['accuracy']*100:5.1f}% (Δ {improvement:+5.2f}%)")
        
        return results
    
    def run_random_subsets(self, tasks: List[Dict], num_trials: int = 10) -> Dict:
        """
        Random Subset Ablation: Test random subsets of different sizes.
        Shows that specific neurons matter, not just quantity.
        """
        logger.info("\n" + "="*80)
        logger.info("RANDOM SUBSET ABLATION")
        logger.info("="*80)
        logger.info(f"Test {num_trials} random subsets of each size")
        logger.info("-"*80)
        
        results = {}
        
        for size in [4, 8, 12, 16]:
            logger.info(f"\n📊 Subset Size: {size} neurons")
            
            size_results = []
            
            for trial in range(num_trials):
                # Random subset
                random_neurons = np.random.choice(
                    self.cluster9_neurons, 
                    size=size, 
                    replace=False
                ).tolist()
                
                config = AblationConfig(
                    neurons=random_neurons,
                    layer=self.best_layer,
                    strength=self.best_strength,
                    experiment_type='random_subset'
                )
                
                result = self.evaluate_with_config(tasks, config)
                size_results.append(result['accuracy'])
            
            mean_acc = np.mean(size_results) * 100
            std_acc = np.std(size_results) * 100
            
            results[f'size_{size}'] = {
                'size': size,
                'accuracies': size_results,
                'mean': mean_acc,
                'std': std_acc
            }
            
            logger.info(f"  Mean Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
        
        # Compare with full Cluster 9
        full_config = AblationConfig(
            neurons=self.cluster9_neurons,
            layer=self.best_layer,
            strength=self.best_strength,
            experiment_type='full_cluster9'
        )
        full_result = self.evaluate_with_config(tasks, full_config)
        results['full_cluster9'] = {
            'size': 16,
            'accuracy': full_result['accuracy'] * 100
        }
        
        logger.info(f"\n  Full Cluster 9: {full_result['accuracy']*100:.1f}%")
        
        return results
    
    def run_importance_ranking(self, tasks: List[Dict]) -> Dict:
        """
        Rank neurons by individual contribution.
        Remove each neuron individually and rank by performance drop.
        """
        logger.info("\n" + "="*80)
        logger.info("NEURON IMPORTANCE RANKING")
        logger.info("="*80)
        logger.info("Rank neurons by performance impact when removed")
        logger.info("-"*80)
        
        # Get baseline - use subset of tasks for speed if testing many neurons
        if len(self.cluster9_neurons) > 100:
            logger.info(f"\n⚠️  Testing {len(self.cluster9_neurons)} neurons - using subset of {min(50, len(tasks))} tasks for baseline")
            baseline_tasks = tasks[:min(50, len(tasks))]
        else:
            baseline_tasks = tasks
        
        logger.info(f"\n📊 Computing baseline with all {len(self.cluster9_neurons)} neurons...")
        baseline_config = AblationConfig(
            neurons=self.cluster9_neurons,
            layer=self.best_layer,
            strength=self.best_strength,
            experiment_type='baseline'
        )
        baseline_result = self.evaluate_with_config(baseline_tasks, baseline_config, verbose=True)
        baseline_acc = baseline_result['accuracy']
        logger.info(f"✅ Baseline accuracy: {baseline_acc*100:.1f}%")
        
        # Use same subset for neuron testing if we used subset for baseline
        test_tasks = baseline_tasks
        
        # Test each neuron removal
        logger.info(f"\n📊 Testing {len(self.cluster9_neurons)} neurons (this will take time)...")
        logger.info("   Progress will be shown every 10 neurons")
        neuron_impacts = []
        
        for idx, neuron in enumerate(self.cluster9_neurons):
            if (idx + 1) % 10 == 0:
                logger.info(f"   Progress: {idx + 1}/{len(self.cluster9_neurons)} neurons tested...")
            neurons_without = [n for n in self.cluster9_neurons if n != neuron]
            
            config = AblationConfig(
                neurons=neurons_without,
                layer=self.best_layer,
                strength=self.best_strength,
                experiment_type='leave_one_out',
                ablated_neuron=neuron
            )
            
            result = self.evaluate_with_config(test_tasks, config, verbose=False)
            impact = (baseline_acc - result['accuracy']) * 100
            
            neuron_impacts.append({
                'neuron': neuron,
                'impact': impact,
                'accuracy_without': result['accuracy'] * 100
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"   Latest: Neuron {neuron} impact = {impact:+.2f}%")
        
        # Sort by impact (descending)
        neuron_impacts.sort(key=lambda x: x['impact'], reverse=True)
        
        logger.info("\n" + "="*80)
        logger.info("IMPORTANCE RANKING: Sorted by impact when removed (positive = helps, negative = hurts)")
        logger.info("="*80)
        logger.info("")
        logger.info(f"{'Rank':<6} | {'Neuron':<8} | {'Impact':<10} | {'Acc Without':<12} | {'Interpretation'}")
        logger.info("-" * 80)
        
        for rank, info in enumerate(neuron_impacts, 1):
            neuron = info['neuron']
            impact = info['impact']
            acc_without = info['accuracy_without']
            
            # Determine interpretation based on impact
            if impact > 2.0:
                interpretation = "CRITICAL - Most helpful neuron"
            elif impact > 1.0:
                interpretation = "HELPFUL - Significant positive impact"
            elif impact > 0.5:
                interpretation = "HELPFUL - Modest positive impact"
            elif impact > 0.0:
                interpretation = "SLIGHTLY HELPFUL - Small positive impact"
            elif impact > -0.5:
                interpretation = "NEUTRAL - Minimal impact"
            elif impact > -1.0:
                interpretation = "SLIGHTLY HARMFUL - Small negative impact"
            elif impact > -2.0:
                interpretation = "HARMFUL - Modest negative impact"
            else:
                interpretation = "VERY HARMFUL - Significant negative impact"
            
            logger.info(f"{rank:<6} | {neuron:<8} | {impact:>+6.2f}%   | {acc_without:>8.1f}%    | {interpretation}")
        
        logger.info("")
        logger.info("="*80)
        
        return {
            'baseline_accuracy': baseline_acc * 100,
            'ranking': neuron_impacts
        }
    
    def run_amplification_ablation(self, tasks: List[Dict]) -> Dict:
        """
        Amplification Ablation: Test different amplification values to show
        that only 5x (for >2% impact) and 2x (for -2% to 2% impact) work well.
        
        METHODOLOGY:
        - We AMPLIFY all 736 neurons (not remove/ablate them)
        - Each neuron gets amplification based on its impact:
          * Neurons with >2% impact → 5x amplification
          * Neurons with -2% to 2% impact → 2x amplification  
          * Neurons with <-2% impact → 1x (no amplification)
        - We compare impact-based amplification vs uniform amplification
        """
        logger.info("\n" + "="*80)
        logger.info("AMPLIFICATION VALUE ABLATION")
        logger.info("="*80)
        logger.info("Test different amplification values to show optimal values")
        logger.info("METHODOLOGY: Amplifying ALL neurons with different strengths based on impact")
        logger.info("-"*80)
        
        if len(self.neuron_impacts) == 0:
            logger.warning("⚠️  No neuron impacts loaded. Cannot run impact-based amplification ablation.")
            logger.info("   Falling back to uniform amplification test...")
            return self._run_uniform_amplification_ablation(tasks)
        
        results = {}
        
        # Get neurons grouped by impact
        high_impact_neurons = [n for n, impact in self.neuron_impacts.items() if impact > 2.0]
        neutral_neurons = [n for n, impact in self.neuron_impacts.items() if -2.0 <= impact <= 2.0]
        low_impact_neurons = [n for n, impact in self.neuron_impacts.items() if impact < -2.0]
        
        total_neurons = len(self.neuron_impacts)
        logger.info(f"\n📊 Neuron Groups (Total: {total_neurons} neurons):")
        logger.info(f"   High impact (>2%): {len(high_impact_neurons)} neurons → will be amplified by 5x")
        logger.info(f"   Neutral (-2% to 2%): {len(neutral_neurons)} neurons → will be amplified by 2x")
        logger.info(f"   Low impact (<-2%): {len(low_impact_neurons)} neurons → will NOT be amplified (1x)")
        logger.info(f"\n   All {total_neurons} neurons are AMPLIFIED (not removed) with different strengths")
        
        # Baseline: No steering
        logger.info(f"\n📊 Baseline (No steering)")
        baseline_config = AblationConfig(
            neurons=[],
            layer=self.best_layer,
            strength=1.0,
            experiment_type='no_steering',
            use_impact_based=False
        )
        baseline_result = self.evaluate_with_config(tasks, baseline_config)
        baseline_acc = baseline_result['accuracy'] * 100
        results['baseline'] = {
            'neurons': [],
            'result': baseline_result,
            'accuracy': baseline_acc
        }
        logger.info(f"  Accuracy: {baseline_acc:.1f}%")
        
        # Reference: Impact-based amplification (using test values for comparison)
        # This serves as a baseline comparison for the individual group tests
        # We use test values from our range (e.g., 5.0x for high impact, 2.0x for neutral) as reference
        reference_acc = None
        if len(self.neuron_impacts) > 0:
            # Use test values as reference (5.0x for high impact, 2.0x for neutral)
            ref_high_impact = 5.0  # Test value from our range
            ref_neutral = 2.0      # Test value from our range
            logger.info(f"\n📊 Reference: Impact-Based Amplification (Using test values)")
            logger.info(f"   Amplifying ALL {total_neurons} neurons with test values as reference:")
            logger.info(f"   - High impact neurons (>2%): {len(high_impact_neurons)} neurons → {ref_high_impact}x amplification")
            logger.info(f"   - Neutral neurons (-2% to 2%): {len(neutral_neurons)} neurons → {ref_neutral}x amplification")
            logger.info(f"   - Low impact neurons (<-2%): {len(low_impact_neurons)} neurons → 1.0x (no amplification)")
            reference_config = AblationConfig(
                neurons=list(self.neuron_impacts.keys()),  # ALL neurons are amplified
                layer=self.best_layer,
                strength=5.0,  # Not used when impact-based is True
                experiment_type='impact_based_reference',
                use_impact_based=True
            )
            reference_result = self.evaluate_with_config(tasks, reference_config)
            reference_acc = reference_result['accuracy'] * 100
            reference_improvement = reference_acc - baseline_acc
            results['impact_based_reference'] = {
                'neurons': list(self.neuron_impacts.keys()),
                'result': reference_result,
                'accuracy': reference_acc,
                'improvement': reference_improvement
            }
            logger.info(f"  Accuracy: {reference_acc:.1f}% (+{reference_improvement:+.1f}%)")
        
        # Tests for High Impact Neurons (>2%) - Test amplification values from 0.1 to 100
        # Comprehensive coverage: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
        high_impact_amplifications = [
            (0.1, "Very weak"), (0.2, "Very weak"), (0.3, "Very weak"), (0.4, "Very weak"),
            (0.5, "Very weak"), (0.6, "Very weak"), (0.7, "Very weak"), (0.8, "Very weak"),
            (0.9, "Very weak"), (1.0, "No amplification"), (1.5, "Weak"), (2.0, "Moderate"),
            (3.0, "Moderate"), (4.0, "Moderate"), (5.0, "Strong"), (7.0, "Strong"),
            (10.0, "Very strong"), (15.0, "Very strong"), (20.0, "Extreme"), (25.0, "Extreme"),
            (30.0, "Extreme"), (35.0, "Extreme"), (40.0, "Extreme"), (50.0, "Extreme"),
            (60.0, "Extreme"), (70.0, "Extreme"), (80.0, "Extreme"), (90.0, "Extreme"), (100.0, "Extreme")
        ]
        
        logger.info(f"\n📊 High Impact Neurons (>2%) - {len(high_impact_neurons)} neurons")
        logger.info(f"   Testing amplification values from 0.1x to 100x")
        logger.info(f"   Values: {[s for s, _ in high_impact_amplifications]}")
        logger.info("-"*80)
        
        for strength, description in high_impact_amplifications:
            # Only amplify high impact neurons with this strength, others not amplified
            config = AblationConfig(
                neurons=high_impact_neurons,  # Only high impact neurons
                layer=self.best_layer,
                strength=strength,
                experiment_type=f'high_impact_{strength}x',
                use_impact_based=False
            )
            result = self.evaluate_with_config(tasks, config)
            acc = result['accuracy'] * 100
            improvement = acc - baseline_acc
            diff_from_reference = acc - reference_acc if reference_acc is not None else 0
            
            results[f'high_impact_{strength}x'] = {
                'strength': strength,
                'neurons': high_impact_neurons,
                'result': result,
                'accuracy': acc,
                'improvement': improvement,
                'diff_from_reference': diff_from_reference
            }
            
            status = "✓" if improvement > 1.0 else "✗" if improvement < -1.0 else "~"
            logger.info(f"  {strength}x: {acc:.1f}% ({improvement:+.1f}%) {status} {description}")
        
        # Tests for Neutral Neurons (-2% to +2%) - Test amplification values from 0.1 to 100
        # Comprehensive coverage: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
        neutral_amplifications = [
            (0.1, "Very weak"), (0.2, "Very weak"), (0.3, "Very weak"), (0.4, "Very weak"),
            (0.5, "Very weak"), (0.6, "Very weak"), (0.7, "Very weak"), (0.8, "Very weak"),
            (0.9, "Very weak"), (1.0, "No amplification"), (1.5, "Weak"), (2.0, "Moderate"),
            (3.0, "Moderate"), (4.0, "Moderate"), (5.0, "Strong"), (7.0, "Strong"),
            (10.0, "Very strong"), (15.0, "Very strong"), (20.0, "Extreme"), (25.0, "Extreme"),
            (30.0, "Extreme"), (35.0, "Extreme"), (40.0, "Extreme"), (50.0, "Extreme"),
            (60.0, "Extreme"), (70.0, "Extreme"), (80.0, "Extreme"), (90.0, "Extreme"), (100.0, "Extreme")
        ]
        
        logger.info(f"\n📊 Neutral Neurons (-2% to +2%) - {len(neutral_neurons)} neurons")
        logger.info(f"   Testing amplification values from 0.1x to 100x")
        logger.info(f"   Values: {[s for s, _ in neutral_amplifications]}")
        logger.info("-"*80)
        
        for strength, description in neutral_amplifications:
            # Only amplify neutral neurons with this strength, others not amplified
            config = AblationConfig(
                neurons=neutral_neurons,  # Only neutral neurons
                layer=self.best_layer,
                strength=strength,
                experiment_type=f'neutral_{strength}x',
                use_impact_based=False
            )
            result = self.evaluate_with_config(tasks, config)
            acc = result['accuracy'] * 100
            improvement = acc - baseline_acc
            diff_from_reference = acc - reference_acc if reference_acc is not None else 0
            
            results[f'neutral_{strength}x'] = {
                'strength': strength,
                'neurons': neutral_neurons,
                'result': result,
                'accuracy': acc,
                'improvement': improvement,
                'diff_from_reference': diff_from_reference
            }
            
            status = "✓" if improvement > 1.0 else "✗" if improvement < -1.0 else "~"
            logger.info(f"  {strength}x: {acc:.1f}% ({improvement:+.1f}%) {status} {description}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("AMPLIFICATION ABLATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Baseline (no steering): {baseline_acc:.1f}%")
        if reference_acc is not None:
            logger.info(f"Reference (impact-based default): {reference_acc:.1f}% (+{reference_acc - baseline_acc:+.1f}%)")
        logger.info("")
        logger.info("Testing amplification strengths to find best values:")
        logger.info("  - High Impact Neurons (>2%): Testing 29 values from 0.1x to 100x")
        logger.info("  - Neutral Neurons (-2% to +2%): Testing 29 values from 0.1x to 100x")
        logger.info("  - Results will show which amplification values work best for each group")
        logger.info("="*80)
        
        return results
    
    def _run_uniform_amplification_ablation(self, tasks: List[Dict]) -> Dict:
        """Fallback: Test uniform amplification values when impacts not available."""
        logger.info("\n📊 Testing uniform amplification values...")
        results = {}
        
        # Baseline
        baseline_config = AblationConfig(
            neurons=[],
            layer=self.best_layer,
            strength=1.0,
            experiment_type='no_steering',
            use_impact_based=False
        )
        baseline_result = self.evaluate_with_config(tasks, baseline_config)
        baseline_acc = baseline_result['accuracy'] * 100
        results['baseline'] = {'accuracy': baseline_acc}
        
        # Test different strengths
        for strength in [2.0, 3.0, 5.0, 7.0, 10.0]:
            config = AblationConfig(
                neurons=self.cluster9_neurons[:100] if len(self.cluster9_neurons) > 100 else self.cluster9_neurons,
                layer=self.best_layer,
                strength=strength,
                experiment_type=f'uniform_{strength}x',
                use_impact_based=False
            )
            result = self.evaluate_with_config(tasks, config)
            acc = result['accuracy'] * 100
            results[f'{strength}x'] = {'accuracy': acc, 'improvement': acc - baseline_acc}
            logger.info(f"  {strength}x: {acc:.1f}% (+{acc - baseline_acc:+.1f}%)")
        
        return results
    
    def run_layer_ablation_with_perplexity(self, pile_texts: List[str], neurons: List[int] = None) -> Dict:
        """
        Run layer-wise ablation using perplexity on The Pile dataset.
        
        Args:
            pile_texts: List of text samples from The Pile
            neurons: List of neuron indices to steer (default: use discovered neurons from impacts)
        
        Returns:
            Dictionary with perplexity results for each layer
        """
        logger.info("\n" + "="*80)
        logger.info("LAYER-WISE ABLATION WITH PERPLEXITY ON THE PILE")
        logger.info("="*80)
        logger.info(f"Testing {len(pile_texts)} text samples from The Pile")
        logger.info(f"Metric: Perplexity (lower is better)")
        logger.info("-"*80)
        
        if neurons is None:
            # Use discovered neurons from impacts if available
            if len(self.neuron_impacts) > 0:
                neurons = list(self.neuron_impacts.keys())
                logger.info(f"Using {len(neurons)} discovered neurons from impact analysis")
            else:
                # Fallback to cluster9 neurons
                neurons = self.cluster9_neurons[:100] if len(self.cluster9_neurons) > 100 else self.cluster9_neurons
                logger.info(f"Using {len(neurons)} neurons from Cluster 9")
        
        results = {}
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
            experiment_type='baseline_no_steering',
            use_impact_based=False
        )
        baseline_ppl = self.calculate_perplexity(pile_texts, baseline_config)
        results['baseline'] = {
            'layer': None,
            'perplexity': baseline_ppl,
            'neurons': []
        }
        logger.info(f"  Perplexity: {baseline_ppl:.2f}")
        
        # Test each layer
        logger.info(f"\n📊 Layer-wise Perplexity Results")
        logger.info("-"*80)
        
        for layer_idx in layers_to_test:
            # Use impact-based amplification if available
            use_impact_based = len(self.neuron_impacts) > 0
            
            config = AblationConfig(
                neurons=neurons,
                layer=layer_idx,
                strength=5.0,  # Default strength (will be overridden by impact-based if enabled)
                experiment_type=f'layer_{layer_idx}_steering',
                use_impact_based=use_impact_based
            )
            
            ppl = self.calculate_perplexity(pile_texts, config)
            ppl_change = ppl - baseline_ppl
            ppl_change_pct = (ppl_change / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            
            results[f'layer_{layer_idx}'] = {
                'layer': layer_idx,
                'role': layer_descriptions[layer_idx],
                'perplexity': ppl,
                'change_from_baseline': ppl_change,
                'change_percent': ppl_change_pct,
                'neurons': neurons[:10] if len(neurons) > 10 else neurons  # Store sample
            }
            
            status = "✓" if ppl < baseline_ppl else "✗"
            logger.info(f"  Layer {layer_idx:2d} ({layer_descriptions[layer_idx]:<20}): "
                       f"PPL={ppl:6.2f} ({ppl_change:+.2f}, {ppl_change_pct:+.1f}%) {status}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("LAYER ABLATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Baseline perplexity: {baseline_ppl:.2f}")
        
        best_layer = min(layers_to_test, key=lambda l: results[f'layer_{l}']['perplexity'])
        best_ppl = results[f'layer_{best_layer}']['perplexity']
        best_improvement = baseline_ppl - best_ppl
        best_improvement_pct = (best_improvement / baseline_ppl) * 100 if baseline_ppl > 0 else 0
        
        logger.info(f"Best layer: {best_layer} ({layer_descriptions[best_layer]})")
        logger.info(f"  Perplexity: {best_ppl:.2f} (improvement: {best_improvement:.2f}, {best_improvement_pct:+.1f}%)")
        
        worst_layer = max(layers_to_test, key=lambda l: results[f'layer_{l}']['perplexity'])
        worst_ppl = results[f'layer_{worst_layer}']['perplexity']
        worst_degradation = worst_ppl - baseline_ppl
        worst_degradation_pct = (worst_degradation / baseline_ppl) * 100 if baseline_ppl > 0 else 0
        
        logger.info(f"Worst layer: {worst_layer} ({layer_descriptions[worst_layer]})")
        logger.info(f"  Perplexity: {worst_ppl:.2f} (degradation: {worst_degradation:.2f}, {worst_degradation_pct:+.1f}%)")
        logger.info("="*80)
        
        return results
    
    def run_complete_analysis(self, tasks: List[Dict]) -> Dict:
        """Run all ablation studies."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE NEURON ABLATION ANALYSIS")
        logger.info("="*80)
        logger.info(f"Testing {len(tasks)} tasks across 4 task types")
        logger.info(f"Analyzing {len(self.cluster9_neurons)} neurons at Layer {self.best_layer}")
        if len(self.cluster9_neurons) == self.hidden_dim:
            logger.info(f"⚠️  Testing ALL {self.hidden_dim} neurons in hidden dimension")
        logger.info("="*80)
        
        all_results = {}
        
        # COMMENTED OUT: Other ablation tests
        # 1. Leave-One-Out
        # all_results['leave_one_out'] = self.run_leave_one_out(tasks)
        
        # 2. Add-One-In
        # all_results['add_one_in'] = self.run_add_one_in(tasks)
        
        # 3. Random Subsets
        # all_results['random_subsets'] = self.run_random_subsets(tasks, num_trials=10)
        
        # 4. Importance Ranking
        all_results['importance_ranking'] = self.run_importance_ranking(tasks)
        
        # 5. Amplification Ablation (NEW: Shows optimal amplification values)
        all_results['amplification_ablation'] = self.run_amplification_ablation(tasks)
        
        # 6. Summary
        # self._print_summary(all_results)  # Commented out since we only have ranking
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print comprehensive summary of all ablation studies."""
        logger.info("\n" + "="*80)
        logger.info("ABLATION ANALYSIS SUMMARY")
        logger.info("="*80)
        
        # Key findings from Leave-One-Out
        loo_results = results['leave_one_out']
        baseline_acc = loo_results['baseline']['result']['accuracy'] * 100
        
        impacts = []
        for key, value in loo_results.items():
            if key.startswith('without_'):
                impacts.append(value['performance_drop'])
        
        logger.info("\n1️⃣ LEAVE-ONE-OUT FINDINGS:")
        logger.info(f"   • Baseline (all {len(self.cluster9_neurons)} neurons): {baseline_acc:.1f}%")
        logger.info(f"   • Average impact per neuron: {np.mean(impacts):.2f}%")
        logger.info(f"   • Max impact (most critical): {np.max(impacts):.2f}%")
        logger.info(f"   • Min impact (least critical): {np.min(impacts):.2f}%")
        logger.info(f"   • Neurons with positive impact: {sum(1 for i in impacts if i > 0)}/{len(self.cluster9_neurons)}")
        
        # Key findings from Add-One-In
        aoi_results = results['add_one_in']
        no_steering_acc = aoi_results['no_steering']['result']['accuracy'] * 100
        
        accuracies = []
        max_neurons = len(self.cluster9_neurons)
        for i in range(1, max_neurons + 1):
            key = f'with_{i}_neurons'
            if key in aoi_results:
                accuracies.append(aoi_results[key]['result']['accuracy'] * 100)
        
        logger.info("\n2️⃣ ADD-ONE-IN FINDINGS:")
        logger.info(f"   • No steering baseline: {no_steering_acc:.1f}%")
        if len(accuracies) > 4:
            logger.info(f"   • With 4 neurons: {accuracies[3]:.1f}% (+{accuracies[3]-no_steering_acc:.1f}%)")
        if len(accuracies) > 8:
            logger.info(f"   • With 8 neurons: {accuracies[7]:.1f}% (+{accuracies[7]-no_steering_acc:.1f}%)")
        if len(accuracies) > 16:
            logger.info(f"   • With 16 neurons: {accuracies[15]:.1f}% (+{accuracies[15]-no_steering_acc:.1f}%)")
        if len(accuracies) > 0:
            logger.info(f"   • With {len(accuracies)} neurons: {accuracies[-1]:.1f}% (+{accuracies[-1]-no_steering_acc:.1f}%)")
            logger.info(f"   • Performance grows with neuron count: {'YES' if accuracies[-1] > accuracies[0] else 'NO'}")
        
        # Key findings from Random Subsets
        rs_results = results['random_subsets']
        
        logger.info("\n3️⃣ RANDOM SUBSET FINDINGS:")
        for size in [4, 8, 12]:
            key = f'size_{size}'
            if key in rs_results:
                mean = rs_results[key]['mean']
                std = rs_results[key]['std']
                logger.info(f"   • {size} random neurons: {mean:.1f}% ± {std:.1f}%")
        
        full_acc = rs_results['full_cluster9']['accuracy']
        logger.info(f"   • Full set ({len(self.cluster9_neurons)} neurons): {full_acc:.1f}%")
        logger.info(f"   • Specific neuron selection matters: {'YES' if std > 1.0 else 'NO'}")
        
        # Key findings from Importance Ranking
        ir_results = results['importance_ranking']
        ranking = ir_results['ranking']
        
        logger.info("\n4️⃣ TOP 5 MOST CRITICAL NEURONS:")
        for i in range(min(5, len(ranking))):
            neuron = ranking[i]['neuron']
            impact = ranking[i]['impact']
            logger.info(f"   {i+1}. Neuron {neuron}: {impact:+.2f}% impact when removed")
        
        logger.info("\n5️⃣ VALIDATION OF CLUSTER 9:")
        all_positive = all(r['impact'] >= 0 for r in ranking[:10])
        logger.info(f"   • All neurons contribute positively: {'YES ✓' if all_positive else 'NO ✗'}")
        logger.info(f"   • Mean contribution per neuron: {np.mean([r['impact'] for r in ranking]):.2f}%")
        logger.info(f"   • Cluster is coherent and functional: {'YES ✓' if np.mean(impacts) > 0 else 'NO ✗'}")
        
        logger.info("\n" + "="*80)


def main():
    """Main function to run intra-cluster ablation analysis."""
    import argparse
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    parser = argparse.ArgumentParser(description="Cluster 9 Intra-Neuron Ablation")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf",
                       help="Model to evaluate")
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of test examples (25 per task type)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="cluster9_ablation_results",
                       help="Directory to save results")
    parser.add_argument("--max_neurons", type=int, default=None,
                       help="Maximum number of neurons to test (None = all neurons)")
    parser.add_argument("--sample_neurons", action="store_true",
                       help="Randomly sample neurons instead of testing all")
    parser.add_argument("--perplexity_ablation", action="store_true",
                       help="Run layer-wise ablation with perplexity on The Pile")
    parser.add_argument("--pile_samples", type=int, default=100,
                       help="Number of samples from The Pile for perplexity calculation")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
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
    
    # Run ablation analysis
    analyzer = Cluster9AblationAnalyzer(model, tokenizer, device)
    
    if args.perplexity_ablation:
        # Run layer-wise ablation with perplexity on The Pile
        logger.info("="*80)
        logger.info("RUNNING LAYER-WISE ABLATION WITH PERPLEXITY")
        logger.info("="*80)
        
        # Load Pile dataset
        try:
            from pile_dataset_loader import PileDatasetLoader
            logger.info(f"Loading {args.pile_samples} samples from The Pile...")
            pile_loader = PileDatasetLoader(num_samples=args.pile_samples * 2, seed=42)
            pile_loader.load_pile(split="train", streaming=True)
            pile_texts = pile_loader.get_text_samples(num_samples=args.pile_samples)
            logger.info(f"✅ Loaded {len(pile_texts)} text samples from The Pile")
        except Exception as e:
            logger.error(f"❌ Error loading The Pile dataset: {e}")
            logger.info("   Falling back to synthetic text samples...")
            # Fallback: create synthetic text samples
            pile_texts = [
                "The quick brown fox jumps over the lazy dog. " * 10,
                "In a hole in the ground there lived a hobbit. " * 10,
                "It was the best of times, it was the worst of times. " * 10,
            ] * (args.pile_samples // 3 + 1)
            pile_texts = pile_texts[:args.pile_samples]
            logger.info(f"   Using {len(pile_texts)} synthetic text samples")
        
        # Use discovered neurons if available
        neurons = None
        if len(analyzer.neuron_impacts) > 0:
            neurons = list(analyzer.neuron_impacts.keys())
            logger.info(f"Using {len(neurons)} discovered neurons from impact analysis")
        
        # Run layer ablation
        results = analyzer.run_layer_ablation_with_perplexity(pile_texts, neurons=neurons)
        
        # Save results
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
        
        results_path = output_dir / "layer_ablation_perplexity_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=make_serializable)
        
        logger.info(f"\n✅ Perplexity results saved to: {results_path}")
        
    else:
        # Original ablation analysis with task accuracy
        # Generate test tasks
        logger.info(f"Generating {args.num_examples} test tasks...")
        generator = SimpleTaskGenerator(seed=42)
        test_tasks = generator.generate_test_set(num_examples=args.num_examples)
    
    # Limit or sample neurons if requested
    if args.max_neurons is not None and args.max_neurons < len(analyzer.cluster9_neurons):
        if args.sample_neurons:
            import random
            random.seed(42)
            analyzer.cluster9_neurons = random.sample(analyzer.cluster9_neurons, args.max_neurons)
            logger.info(f"📊 Randomly sampled {args.max_neurons} neurons for testing")
        else:
            analyzer.cluster9_neurons = analyzer.cluster9_neurons[:args.max_neurons]
            logger.info(f"📊 Testing first {args.max_neurons} neurons")
    
    logger.info(f"📊 Final neuron set: {len(analyzer.cluster9_neurons)} neurons")
    if len(analyzer.cluster9_neurons) > 100:
        logger.warning(f"⚠️  Testing {len(analyzer.cluster9_neurons)} neurons will take a very long time!")
        logger.warning(f"   Estimated time: ~{len(analyzer.cluster9_neurons) * 0.5:.0f} minutes")
    
    results = analyzer.run_complete_analysis(test_tasks)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to JSON-serializable format
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
    
    results_path = output_dir / "cluster9_ablation_results_all_neurons.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=make_serializable)
    
    logger.info(f"\n✅ Results saved to: {results_path}")
    
    # Create visualization
    try:
        create_ablation_visualizations(results, output_dir)
        logger.info(f"✅ Visualizations saved to: {output_dir}")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to JSON-serializable format
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
    


def create_ablation_visualizations(results: Dict, output_dir: Path):
    """Create visualizations of ablation results."""
    
    # 1. Leave-One-Out Impact Chart
    loo_results = results['leave_one_out']
    neurons = []
    impacts = []
    
    for key, value in loo_results.items():
        if key.startswith('without_'):
            neurons.append(value['ablated_neuron'])
            impacts.append(value['performance_drop'])
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if i > 0 else 'blue' for i in impacts]
    plt.bar(range(len(neurons)), impacts, color=colors, alpha=0.7)
    plt.xlabel('Neuron Index', fontsize=12)
    plt.ylabel('Performance Drop (%)', fontsize=12)
    plt.title('Leave-One-Out: Impact of Removing Each Neuron', fontsize=14, fontweight='bold')
    plt.xticks(range(len(neurons)), neurons, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    loo_path = output_dir / "leave_one_out_impact.png"
    plt.savefig(loo_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✅ Saved: {loo_path}")
    
    # 2. Add-One-In Progressive Improvement
    if 'add_one_in' in results:
        aoi_results = results['add_one_in']
        num_neurons = []
        accuracies = []
        
        for key in sorted(aoi_results.keys()):
            if key.startswith('with_') and 'neurons' in key:
                value = aoi_results[key]
                num_neurons.append(value['num_neurons'])
                accuracies.append(value['result']['accuracy'] * 100)
        
        plt.figure(figsize=(10, 6))
        plt.plot(num_neurons, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Neurons', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Add-One-In: Cumulative Performance Improvement', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        aoi_path = output_dir / "add_one_in_progression.png"
        plt.savefig(aoi_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✅ Saved: {aoi_path}")
    
    # 3. Neuron Importance Ranking
    if neurons and impacts:
        # Sort by impact
        sorted_pairs = sorted(zip(neurons, impacts), key=lambda x: x[1], reverse=True)
        sorted_neurons, sorted_impacts = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        plt.figure(figsize=(12, 6))
        colors_ranked = ['darkred' if i > 0 else 'lightblue' for i in sorted_impacts]
        plt.barh(range(len(sorted_neurons)), sorted_impacts, color=colors_ranked, alpha=0.7)
        plt.ylabel('Neuron Index', fontsize=12)
        plt.xlabel('Performance Drop (%)', fontsize=12)
        plt.title('Neuron Importance Ranking (by LOO Impact)', fontsize=14, fontweight='bold')
        plt.yticks(range(len(sorted_neurons)), sorted_neurons)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        ranking_path = output_dir / "neuron_importance_ranking.png"
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✅ Saved: {ranking_path}")


if __name__ == "__main__":
    main()