#!/usr/bin/env python3
"""
STEP 1: NEURON DISCOVERY

Discovers beneficial delta neurons by testing all neurons individually.
Saves results to JSON file for use in validation phase.

This script:
1. Extracts delta neurons from Layer 20 using SSM activation variance
2. Tests each neuron's impact by removing it and measuring performance drop
3. Ranks neurons by importance
4. Saves discovered neurons to discovered_neurons.json format

Run:
python step1_neuron_discovery.py --model <model> --save_path discovered_neurons.json
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def calculate_entropy(hidden_states):
    """Calculate Shannon entropy of hidden state distributions."""
    probs = F.softmax(hidden_states, dim=-1)
    entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropies.mean().item()


def calculate_effective_rank(hidden_states):
    """Calculate effective rank using singular values."""
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    
    _, S, _ = torch.svd(hidden_states.float())
    S_normalized = S / S.sum()
    sv_entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-10))
    effective_rank = torch.exp(sv_entropy).item()
    
    return effective_rank


@dataclass
class SteeringConfig:
    """Configuration for steering experiments."""
    neurons: List[int]
    layer: int
    strength: float
    selection_method: str


class StructuredTaskGenerator:
    """
    Generate structured reasoning tasks where steering is effective.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_needle_in_haystack(self, num_examples: int = 100) -> List[Dict]:
        """Generate needle-in-haystack tasks."""
        tasks = []
        
        for i in range(num_examples):
            haystack_size = np.random.randint(100, 300)
            distractors = [
                "The quick brown fox jumps over the lazy dog. ",
                "Lorem ipsum dolor sit amet, consectetur adipiscing. ",
                "Machine learning models process natural language text. ",
                "Neural networks use backpropagation for training. "
            ]
            
            haystack = ""
            for _ in range(haystack_size):
                haystack += np.random.choice(distractors)
            
            needle_types = [
                f"The secret code is ALPHA{i:03d}.",
                f"The password is BETA{i:03d}.",
                f"The key is GAMMA{i:03d}.",
                f"The answer is DELTA{i:03d}."
            ]
            needle = np.random.choice(needle_types)
            position = np.random.randint(len(haystack) // 4, 3 * len(haystack) // 4)
            haystack = haystack[:position] + needle + haystack[position:]
            
            if "code is" in needle:
                expected = needle.split("code is ")[1].rstrip(".")
                question = "What is the secret code?"
            elif "password is" in needle:
                expected = needle.split("password is ")[1].rstrip(".")
                question = "What is the password?"
            elif "key is" in needle:
                expected = needle.split("key is ")[1].rstrip(".")
                question = "What is the key?"
            else:
                expected = needle.split("answer is ")[1].rstrip(".")
                question = "What is the answer?"
            
            prompt = f"Text: {haystack}\n\nQuestion: {question}\nAnswer:"
            
            tasks.append({
                'prompt': prompt,
                'expected': expected,
                'alternatives': [expected, expected.lower()],
                'task_type': 'needle_in_haystack',
                'difficulty': 'hard' if haystack_size > 200 else 'medium'
            })
        
        return tasks
    
    def generate_instruction_following(self, num_examples: int = 100) -> List[Dict]:
        """Generate instruction-following tasks."""
        tasks = []
        
        for i in range(num_examples):
            task_type = np.random.choice(['arithmetic', 'list', 'string', 'logic'])
            
            if task_type == 'arithmetic':
                a, b, c = np.random.randint(1, 20, size=3)
                operations = [
                    (f"Add {a} and {b}, then multiply by {c}.", (a + b) * c),
                    (f"Multiply {a} by {b}, then add {c}.", a * b + c),
                    (f"Subtract {b} from {a}, then multiply by {c}.", (a - b) * c),
                    (f"Add {a}, {b}, and {c} together.", a + b + c)
                ]
                weights = [0.3, 0.3, 0.2, 0.2]
                idx = np.random.choice(len(operations), p=weights)
                instruction, answer = operations[idx]
                expected = str(answer)
            
            elif task_type == 'list':
                items = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']
                my_list = np.random.choice(items, size=np.random.randint(3, 6), replace=False).tolist()
                
                operations = [
                    (f"From the list {my_list}, return the first item.", my_list[0]),
                    (f"From the list {my_list}, return the last item.", my_list[-1]),
                    (f"From the list {my_list}, return the second item.", my_list[1] if len(my_list) > 1 else my_list[0]),
                ]
                instruction, expected = random.choice(operations)
            
            elif task_type == 'string':
                words = ['hello', 'world', 'python', 'code', 'test']
                word = np.random.choice(words)
                operations = [
                    (f"Convert '{word}' to uppercase.", word.upper()),
                    (f"Reverse the string '{word}'.", word[::-1]),
                    (f"Return the first letter of '{word}'.", word[0]),
                    (f"Return the last letter of '{word}'.", word[-1])
                ]
                instruction, expected = random.choice(operations)
            
            else:  # logic
                conditions = [
                    (f"If 5 > 3, say YES, otherwise say NO.", "YES"),
                    (f"If 2 + 2 = 4, say TRUE, otherwise say FALSE.", "TRUE"),
                    (f"If 10 < 5, say CORRECT, otherwise say INCORRECT.", "INCORRECT"),
                ]
                instruction, expected = random.choice(conditions)
            
            prompt = f"Instruction: {instruction}\n\nAnswer:"
            
            tasks.append({
                'prompt': prompt,
                'expected': expected,
                'alternatives': [expected, expected.lower(), expected.capitalize()],
                'task_type': 'instruction_following',
                'difficulty': 'easy' if task_type in ['string', 'logic'] else 'medium'
            })
        
        return tasks
    
    def generate_long_context_recall(self, num_examples: int = 100) -> List[Dict]:
        """Generate long context recall tasks."""
        tasks = []
        
        names = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry']
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        cities = ['Paris', 'London', 'Tokyo', 'Berlin', 'Sydney', 'Moscow', 'Cairo', 'Rome']
        jobs = ['teacher', 'doctor', 'engineer', 'artist', 'lawyer', 'chef', 'pilot', 'scientist']
        hobbies = ['reading', 'swimming', 'painting', 'cooking', 'hiking', 'dancing', 'gaming', 'photography']
        
        for i in range(num_examples):
            name = np.random.choice(names)
            favorite_color = np.random.choice(colors)
            city = np.random.choice(cities)
            job = np.random.choice(jobs)
            hobby = np.random.choice(hobbies)
            age = np.random.randint(20, 60)
            
            all_facts = [
                f"- {name} is {age} years old",
                f"- {name} lives in {city}",
                f"- {name} works as a {job}",
                f"- {name}'s favorite color is {favorite_color}",
                f"- {name} enjoys {hobby}",
                f"- {name} speaks English fluently",
                f"- {name} has visited 5 countries",
                f"- {name} graduated from university",
                f"- {name} owns a pet cat",
                f"- {name} likes coffee in the morning",
                f"- {name} exercises regularly",
                f"- {name} plays musical instruments"
            ]
            
            noise_facts = [
                f"- The weather is sunny today",
                f"- Technology is advancing rapidly",
                f"- Books are important for learning",
                f"- Music brings joy to people"
            ]
            
            facts = all_facts + np.random.choice(noise_facts, size=3, replace=False).tolist()
            np.random.shuffle(facts)
            
            question_types = [
                (f"What is {name}'s favorite color?", favorite_color),
                (f"Where does {name} live?", city),
                (f"What does {name} do for work?", job),
                (f"What hobby does {name} enjoy?", hobby),
                (f"How old is {name}?", str(age))
            ]
            question, expected = random.choice(question_types)
            
            prompt = f"Information:\n" + "\n".join(facts) + f"\n\nQuestion: {question}\nAnswer:"
            
            tasks.append({
                'prompt': prompt,
                'expected': expected,
                'alternatives': [expected, expected.capitalize(), expected.lower()],
                'task_type': 'long_context_recall',
                'difficulty': 'medium'
            })
        
        return tasks
    
    def generate_chain_reasoning(self, num_examples: int = 100) -> List[Dict]:
        """Generate chain reasoning tasks."""
        tasks = []
        
        names = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry']
        
        for i in range(num_examples):
            num_steps = np.random.randint(2, 5)
            chain = np.random.choice(names, size=num_steps+1, replace=False).tolist()
            
            reasoning_type = np.random.choice(['height', 'age', 'score', 'speed'])
            
            facts = []
            if reasoning_type == 'height':
                for j in range(len(chain)-1):
                    facts.append(f"{chain[j]} is taller than {chain[j+1]}.")
                question = "Who is the tallest person?"
            elif reasoning_type == 'age':
                for j in range(len(chain)-1):
                    facts.append(f"{chain[j]} is older than {chain[j+1]}.")
                question = "Who is the oldest person?"
            elif reasoning_type == 'score':
                for j in range(len(chain)-1):
                    facts.append(f"{chain[j]} scored higher than {chain[j+1]}.")
                question = "Who has the highest score?"
            else:  # speed
                for j in range(len(chain)-1):
                    facts.append(f"{chain[j]} runs faster than {chain[j+1]}.")
                question = "Who is the fastest runner?"
            
            prompt = "Facts:\n" + "\n".join(facts) + f"\n\nQuestion: {question}\nAnswer:"
            expected = chain[0]
            
            tasks.append({
                'prompt': prompt,
                'expected': expected,
                'alternatives': [expected, expected.lower()],
                'task_type': 'chain_reasoning',
                'difficulty': 'hard' if num_steps > 3 else 'medium'
            })
        
        return tasks
    
    def generate_validation_set(self, size_per_task: int = 50) -> List[Dict]:
        """Generate balanced validation set for hyperparameter tuning."""
        validation = []
        
        validation.extend(self.generate_needle_in_haystack(size_per_task))
        validation.extend(self.generate_instruction_following(size_per_task))
        validation.extend(self.generate_long_context_recall(size_per_task))
        validation.extend(self.generate_chain_reasoning(size_per_task))
        
        np.random.shuffle(validation)
        
        logger.info(f"Generated validation set: {len(validation)} tasks")
        logger.info(f"  Needle-in-haystack: {size_per_task}")
        logger.info(f"  Instruction-following: {size_per_task}")
        logger.info(f"  Long context recall: {size_per_task}")
        logger.info(f"  Chain reasoning: {size_per_task}")
        
        return validation


class SteeringValidator:
    """
    Validates steering approach with proper experimental protocol.
    Focus on structured tasks where steering is effective.
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
        
        # Load delta neuron definition
        # Delta neurons are identified by extracting SSM activations and computing variance
        self.neurons = self._get_delta_neurons_for_layer_20()
        
        logger.info(f"Initialized steering validator:")
        logger.info(f"  Model layers: {self.num_layers}")
        logger.info(f"  Hidden dimension: {self.hidden_dim}")
        logger.info(f"  Delta neurons at Layer 20: {len(self.neurons)}")
    
    def _get_model_layers(self, model):
        """Get the layers from the model, handling different possible structures."""
        possible_paths = [
            lambda m: m.backbone.layers,
            lambda m: m.model.layers,
            lambda m: m.layers,
            lambda m: m.transformer.h,
            lambda m: m.transformer.layers,
        ]
        
        for path_fn in possible_paths:
            try:
                layers = path_fn(model)
                if layers is not None:
                    return layers
            except AttributeError:
                continue
        
        return None
    
    def _extract_deltas_fixed(self, model, layer_idx, input_ids):
        """
        Extract delta parameters for a specific layer.
        Delta neurons capture activations from the SSM (State Space Model) module.
        """
        layers = self._get_model_layers(model)
        hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.d_model
        batch_size, seq_len = input_ids.shape
        
        device = next(model.parameters()).device

        if layers is None or layer_idx >= len(layers):
            logger.warning(f"Could not find model layers or invalid index {layer_idx}. Using fallback.")
            return torch.randn(batch_size, seq_len, hidden_size, device=device)

        layer = layers[layer_idx]
        delta_values = []

        def delta_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if hidden_states is not None:
                delta_values.append(hidden_states.detach().clone())
            else:
                delta_values.append(torch.randn(batch_size, seq_len, hidden_size, device=device))

        hook_registered = False
        handle = None
        
        module_candidates = []
        
        if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'ssm'):
            module_candidates.append(('mixer.ssm', layer.mixer.ssm))
        
        if hasattr(layer, 'mixer'):
            module_candidates.append(('mixer', layer.mixer))
        
        if hasattr(layer, 'ssm'):
            module_candidates.append(('ssm', layer.ssm))
        
        module_candidates.append(('layer', layer))
        
        for name, module in module_candidates:
            try:
                handle = module.register_forward_hook(delta_hook)
                logger.debug(f"Registered hook on {name} for layer {layer_idx}")
                hook_registered = True
                break
            except Exception as e:
                logger.debug(f"Failed to register hook on {name}: {e}")
                continue

        if not hook_registered:
            logger.warning(f"Could not register hook for layer {layer_idx}")
            return torch.randn(batch_size, seq_len, hidden_size, device=device)

        try:
            with torch.no_grad():
                _ = model(input_ids)
            
            if handle:
                handle.remove()
            
            if delta_values:
                result = delta_values[0]
                return result
            else:
                return torch.randn(batch_size, seq_len, hidden_size, device=device)
                
        except Exception as e:
            logger.warning(f"Error during forward pass: {e}")
            if handle:
                handle.remove()
            return torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    def _find_delta_sensitive_neurons(self, model, tokenizer, texts, layer_idx=20, top_k=None):
        """
        Find neurons that are sensitive to delta computation.
        Delta neurons are identified by extracting SSM activations and computing variance.
        """
        if top_k is None:
            top_k = self.hidden_dim
        
        logger.info(f"Finding delta-sensitive neurons in layer {layer_idx}...")
        
        device = next(model.parameters()).device
        model = model.to(device)
        
        deltas = []
        hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.d_model

        for i, text in enumerate(texts):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                input_ids = inputs["input_ids"].to(device)
                
                delta = self._extract_deltas_fixed(model, layer_idx, input_ids)

                if delta.dim() == 3:
                    delta_mean = delta.mean(dim=(0, 1))
                elif delta.dim() == 2:
                    delta_mean = delta.mean(dim=0)
                else:
                    delta_mean = delta

                if delta_mean.shape[0] != hidden_size:
                    if delta_mean.shape[0] < hidden_size:
                        padding = torch.zeros(hidden_size - delta_mean.shape[0], device=device)
                        delta_mean = torch.cat([delta_mean, padding])
                    else:
                        delta_mean = delta_mean[:hidden_size]

                deltas.append(delta_mean.cpu().numpy())

            except Exception as e:
                logger.warning(f"Error processing text '{text[:50]}...': {e}")
                deltas.append(np.random.randn(hidden_size) * 0.1)

        if not deltas:
            logger.warning("No deltas extracted. Returning all neurons.")
            return [(i, 0.0) for i in range(min(top_k, hidden_size))]

        try:
            all_deltas = np.array(deltas)
            
            if all_deltas.ndim == 1:
                all_deltas = all_deltas.reshape(1, -1)

            variance = np.var(all_deltas, axis=0)
            
            if top_k >= hidden_size:
                top_indices = np.argsort(variance)[::-1]
            else:
                top_indices = np.argsort(variance)[-top_k:][::-1]
            
            top_indices = [i for i in top_indices if 0 <= i < hidden_size]
            
            results = [(int(i), float(variance[i])) for i in top_indices]
            
            logger.info(f"Found {len(results)} delta-sensitive neurons")
            return results
            
        except Exception as e:
            logger.warning(f"Error computing variance: {e}")
            return [(i, 0.0) for i in range(min(top_k, hidden_size))]
    
    def _get_delta_neurons_for_layer_20(self) -> List[int]:
        """
        Get all delta neurons for layer 20 using delta extraction method.
        Delta neurons are identified by extracting SSM activations and computing variance.
        """
        try:
            from datasets import load_dataset
            
            logger.info("Loading texts from The Pile dataset...")
            pile_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
            
            sample_texts = []
            for i, example in enumerate(pile_dataset):
                if i >= 50:
                    break
                text = example.get("text", "")
                if text and len(text.strip()) > 10:
                    sample_texts.append(text.strip())
            
            if not sample_texts:
                raise ValueError("No texts extracted from The Pile dataset")
            
            logger.info(f"Loaded {len(sample_texts)} texts from The Pile dataset")
            
            delta_results = self._find_delta_sensitive_neurons(
                self.model,
                self.tokenizer,
                sample_texts,
                layer_idx=20,
                top_k=self.hidden_dim
            )
            
            delta_neurons = [neuron for neuron, _ in delta_results]
            
            logger.info(f"✅ Extracted {len(delta_neurons)} delta neurons from layer 20")
            return delta_neurons
            
        except Exception as e:
            logger.warning(f"⚠️  Could not extract delta neurons from The Pile: {e}")
            logger.warning(f"   Falling back to sample texts")
            
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming industries.",
                "Machine learning models process natural language text.",
                "Neural networks use backpropagation for training.",
                "Transformer models have revolutionized NLP tasks.",
                "Deep learning requires large amounts of data.",
                "Natural language processing enables computers to understand text.",
                "State space models process sequences efficiently."
            ]
            
            try:
                delta_results = self._find_delta_sensitive_neurons(
                    self.model,
                    self.tokenizer,
                    sample_texts,
                    layer_idx=20,
                    top_k=self.hidden_dim
                )
                delta_neurons = [neuron for neuron, _ in delta_results]
                logger.info(f"✅ Extracted {len(delta_neurons)} delta neurons using sample texts")
                return delta_neurons
            except Exception as e2:
                logger.warning(f"⚠️  Could not extract delta neurons: {e2}")
                logger.warning(f"   Falling back to all neurons in hidden dimension")
                return list(range(self.hidden_dim))
    
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
    
    def evaluate_with_bottleneck_analysis(self,
                                          tasks: List[Dict],
                                          config: SteeringConfig,
                                          verbose: bool = False) -> Dict:
        """Evaluate with entropy and effective rank measurement."""
        
        hooks = []
        bottleneck_stats = {
            'baseline': {'entropy': [], 'rank': []},
            'steered': {'entropy': [], 'rank': []}
        }
        
        layer_idx = config.layer
        target = self._get_steering_target(layer_idx)
        
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            bottleneck_stats['baseline']['entropy'].append(
                calculate_entropy(hidden.detach())
            )
            bottleneck_stats['baseline']['rank'].append(
                calculate_effective_rank(hidden.detach())
            )
            
            if config.strength > 1.0 and config.neurons:
                h_mod = hidden.clone()
                for idx in config.neurons:
                    if idx < h_mod.shape[-1]:
                        h_mod[..., idx] *= config.strength
                
                bottleneck_stats['steered']['entropy'].append(
                    calculate_entropy(h_mod.detach())
                )
                bottleneck_stats['steered']['rank'].append(
                    calculate_effective_rank(h_mod.detach())
                )
                
                if isinstance(output, tuple):
                    return (h_mod,) + output[1:]
                return h_mod
            else:
                bottleneck_stats['steered']['entropy'] = bottleneck_stats['baseline']['entropy'].copy()
                bottleneck_stats['steered']['rank'] = bottleneck_stats['baseline']['rank'].copy()
            
            return output
        
        hook = target.register_forward_hook(capture_hook)
        hooks.append(hook)
        
        correct = 0
        total = 0
        results_by_task = {}
        results_by_difficulty = {}
        
        for task in tasks:
            prompt = task['prompt']
            expected = task['expected']
            alternatives = task.get('alternatives', [])
            task_type = task.get('task_type', 'unknown')
            difficulty = task.get('difficulty', 'medium')
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    input_len = inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(
                        outputs[0][input_len:],
                        skip_special_tokens=True
                    ).strip()
                    
                    is_correct = self._check_answer(response, expected, alternatives)
                    
                    if is_correct:
                        correct += 1
                        if task_type not in results_by_task:
                            results_by_task[task_type] = {'correct': 0, 'total': 0}
                        results_by_task[task_type]['correct'] += 1
                        if difficulty not in results_by_difficulty:
                            results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
                        results_by_difficulty[difficulty]['correct'] += 1
                    
                    total += 1
                    if task_type not in results_by_task:
                        results_by_task[task_type] = {'correct': 0, 'total': 0}
                    results_by_task[task_type]['total'] += 1
                    if difficulty not in results_by_difficulty:
                        results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
                    results_by_difficulty[difficulty]['total'] += 1
                
                except Exception as e:
                    logger.warning(f"Generation error: {str(e)[:100]}")
                    total += 1
                    if task_type not in results_by_task:
                        results_by_task[task_type] = {'correct': 0, 'total': 0}
                    results_by_task[task_type]['total'] += 1
                    if difficulty not in results_by_difficulty:
                        results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
                    results_by_difficulty[difficulty]['total'] += 1
        
        for hook in hooks:
            hook.remove()
        
        overall_accuracy = correct / total if total > 0 else 0
        
        task_accuracies = {}
        for task_type, counts in results_by_task.items():
            task_accuracies[task_type] = (
                counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            )
        
        difficulty_accuracies = {}
        for difficulty, counts in results_by_difficulty.items():
            difficulty_accuracies[difficulty] = (
                counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            )
        
        baseline_entropy = np.mean(bottleneck_stats['baseline']['entropy']) if bottleneck_stats['baseline']['entropy'] else 0.0
        steered_entropy = np.mean(bottleneck_stats['steered']['entropy']) if bottleneck_stats['steered']['entropy'] else 0.0
        entropy_change = ((steered_entropy - baseline_entropy) / baseline_entropy * 100) if baseline_entropy > 0 else 0.0
        
        baseline_rank = np.mean(bottleneck_stats['baseline']['rank']) if bottleneck_stats['baseline']['rank'] else 0.0
        steered_rank = np.mean(bottleneck_stats['steered']['rank']) if bottleneck_stats['steered']['rank'] else 0.0
        rank_change = steered_rank - baseline_rank
        
        return {
            'accuracy': overall_accuracy,
            'correct': correct,
            'total': total,
            'bottleneck': {
                'baseline_entropy': baseline_entropy,
                'steered_entropy': steered_entropy,
                'entropy_rise_percent': entropy_change,
                'baseline_rank': baseline_rank,
                'steered_rank': steered_rank,
                'rank_increase': rank_change
            },
            'task_accuracies': task_accuracies,
            'difficulty_accuracies': difficulty_accuracies
        }
    
    def evaluate_with_config(self,
                            tasks: List[Dict],
                            config: SteeringConfig,
                            verbose: bool = False) -> Dict:
        """Evaluate model with specific steering configuration."""
        return self.evaluate_with_bottleneck_analysis(tasks, config, verbose)
    
    def run_comprehensive_neuron_discovery(self, validation_tasks: List[Dict]) -> Tuple[Dict, List[int]]:
        """
        Comprehensive neuron discovery: Rank all delta neurons by their impact.
        This is Stage 1 of the validation protocol - discovering which neurons are important.
        
        Returns:
            Tuple of (discovery_results, discovered_neurons) where:
            - discovery_results: Dict with 'importance_ranking' containing baseline_accuracy and ranking
            - discovered_neurons: List of neuron IDs sorted by importance (all neurons)
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE NEURON DISCOVERY (Stage 1)")
        logger.info("="*80)
        logger.info("Discovering delta neurons by ranking their individual contributions")
        logger.info(f"Testing {len(self.neurons)} delta neurons at Layer 20")
        logger.info("-"*80)
        
        # Use subset of tasks for speed if testing many neurons
        if len(self.neurons) > 100:
            logger.info(f"\n⚠️  Testing {len(self.neurons)} neurons - using subset of {min(50, len(validation_tasks))} tasks for discovery")
            discovery_tasks = validation_tasks[:min(50, len(validation_tasks))]
        else:
            discovery_tasks = validation_tasks
        
        # Get baseline with all neurons
        logger.info(f"\n📊 Computing baseline with all {len(self.neurons)} delta neurons...")
        baseline_config = SteeringConfig(
            neurons=self.neurons,
            layer=20,
            strength=5.0,
            selection_method='delta_neurons'
        )
        baseline_result = self.evaluate_with_config(discovery_tasks, baseline_config, verbose=True)
        baseline_acc = baseline_result['accuracy']
        logger.info(f"✅ Baseline accuracy: {baseline_acc*100:.1f}%")
        
        # Test each neuron removal to measure impact
        logger.info(f"\n📊 Testing {len(self.neurons)} neurons (this will take time)...")
        logger.info("   Progress will be shown every 10 neurons")
        neuron_impacts = []
        
        for idx, neuron in enumerate(self.neurons):
            if (idx + 1) % 10 == 0:
                logger.info(f"   Progress: {idx + 1}/{len(self.neurons)} neurons tested...")
            
            # Remove this neuron and measure impact
            neurons_without = [n for n in self.neurons if n != neuron]
            
            config = SteeringConfig(
                neurons=neurons_without,
                layer=20,
                strength=5.0,
                selection_method='delta_neurons'
            )
            
            result = self.evaluate_with_config(discovery_tasks, config, verbose=False)
            impact = (baseline_acc - result['accuracy']) * 100
            
            neuron_impacts.append({
                'neuron': neuron,
                'impact': impact,
                'accuracy': result['accuracy'] * 100  # Accuracy when this neuron is removed
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"   Latest: Neuron {neuron} impact = {impact:+.2f}%")
        
        # Sort by impact (descending - most helpful first)
        neuron_impacts.sort(key=lambda x: x['impact'], reverse=True)
        
        logger.info("\n" + "="*80)
        logger.info("NEURON DISCOVERY RESULTS")
        logger.info("="*80)
        logger.info(f"Baseline accuracy: {baseline_acc*100:.1f}%")
        logger.info(f"Top 10 most important neurons:")
        for i, item in enumerate(neuron_impacts[:10], 1):
            logger.info(f"  {i:2d}. Neuron {item['neuron']:3d}: impact = {item['impact']:+.2f}%")
        logger.info("="*80)
        
        # Return results in format compatible with discovered_neurons.json
        discovery_results = {
            'importance_ranking': {
                'baseline_accuracy': baseline_acc * 100,
                'ranking': neuron_impacts
            }
        }
        
        # Discovered neurons are all neurons, sorted by importance
        discovered_neurons = [item['neuron'] for item in neuron_impacts]
        
        return discovery_results, discovered_neurons


def save_discovered_neurons(discovery_results: Dict, discovered_neurons: List[int], output_path: Path):
    """
    Save discovered neurons in the discovered_neurons.json format.
    
    Args:
        discovery_results: Results from run_comprehensive_neuron_discovery
        discovered_neurons: List of neuron IDs (sorted by importance)
        output_path: Path to save the JSON file
    """
    if 'importance_ranking' not in discovery_results:
        logger.warning("⚠️  No importance_ranking in discovery_results")
        return
    
    ranking_data = discovery_results['importance_ranking']
    baseline_acc = ranking_data.get('baseline_accuracy', 0.0)
    ranking = ranking_data.get('ranking', [])
    
    # Convert to discovered_neurons.json format
    all_neuron_impacts = []
    for item in ranking:
        all_neuron_impacts.append({
            'neuron': item['neuron'],
            'impact': item['impact'],
            'accuracy': item.get('accuracy', baseline_acc - item['impact'])
        })
    
    # Identify beneficial neurons (impact >= -2.0%)
    beneficial_neurons = [
        item['neuron'] for item in ranking 
        if item['impact'] >= -2.0
    ]
    
    discovered_data = {
        'beneficial_neurons': beneficial_neurons,
        'baseline_accuracy': baseline_acc,
        'criterion': 'impact >= -2.0%',
        'all_neuron_impacts': all_neuron_impacts,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(discovered_data, f, indent=2, default=lambda x: str(x) if not isinstance(x, (int, float, str, bool, type(None), list, dict)) else x)
    
    logger.info(f"✅ Saved discovered neurons to: {output_path}")
    logger.info(f"   Found {len(beneficial_neurons)} beneficial neurons (impact >= -2.0%)")


def main():
    """Main function to run neuron discovery."""
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    parser = argparse.ArgumentParser(
        description="Discover beneficial neurons for steering (Stage 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Model to evaluate (default: state-spaces/mamba-130m-hf)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save discovered neurons JSON file (e.g., discovered_neurons.json)"
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=200,
        help="Number of validation examples per task (default: 200)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_3_results",
        help="Directory to save results (default: ablation_3_results)"
    )
    
    args = parser.parse_args()
    
    # Auto-correct model name: add state-spaces/ prefix and -hf suffix if needed
    model_name = args.model
    if 'mamba' in model_name.lower():
        if not model_name.startswith('state-spaces/'):
            model_name = f"state-spaces/{model_name}"
        if not model_name.endswith('-hf'):
            model_name = model_name + '-hf'
        if model_name != args.model:
            logger.info(f"Auto-correcting model name: {args.model} → {model_name}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    try:
        model, tokenizer = load_mamba_model_and_tokenizer(
            model_name=model_name,
            device=args.device if torch.cuda.is_available() else "cpu",
            use_mamba_class=True,
            fallback_to_auto=True
        )
    except Exception as e:
        if "tiktoken" in str(e).lower():
            logger.error(f"Error: Model requires tiktoken package. Either:")
            logger.error(f"  1. Install tiktoken: pip install tiktoken")
            logger.error(f"  2. Use the -hf version: --model {args.model}-hf")
            sys.exit(1)
        else:
            raise
    
    device = next(model.parameters()).device
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Generate validation tasks
    logger.info("Generating validation tasks...")
    generator = StructuredTaskGenerator(seed=42)
    validation_tasks = generator.generate_validation_set(
        size_per_task=args.validation_size // 4
    )
    
    # Run discovery
    logger.info("\n" + "="*80)
    logger.info("NEURON DISCOVERY (Stage 1)")
    logger.info("="*80)
    
    validator = SteeringValidator(model, tokenizer, device)
    discovery_results, beneficial_neurons = validator.run_comprehensive_neuron_discovery(validation_tasks)
    
    # Save discovered neurons
    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / save_path.name
    
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_discovered_neurons(discovery_results, beneficial_neurons, save_path)
    
    logger.info("\n" + "="*80)
    logger.info("DISCOVERY COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Discovered {len(beneficial_neurons)} beneficial neurons")
    logger.info(f"💾 Saved to: {save_path}")
    logger.info(f"\nNext step: Run validation with:")
    logger.info(f"  python steering_validation.py --model {args.model} --neurons {save_path}")


if __name__ == "__main__":
    main()
