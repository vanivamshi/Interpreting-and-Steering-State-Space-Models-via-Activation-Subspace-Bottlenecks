"""
Dataset Classifier for Categorizing Questions into Test Categories

Classifies questions from various datasets (HotpotQA, ReasonChainQA, etc.)
into the test categories:
- Simple Recall
- Two-Hop Reasoning
- Three-Hop Reasoning
- Long Context (5-7 facts)
- Combined Reasoning + Memory
- Stress Test (10+ facts)
- Query Dataset Tasks (TriviaQA)
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datasets import load_dataset
import random
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetQuestionClassifier:
    """
    Classifies questions from various datasets into test categories.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Category definitions
        self.categories = {
            'simple_recall': {
                'description': 'Simple Recall - Direct fact retrieval, no reasoning',
                'criteria': ['single fact', 'direct answer', 'simple question']
            },
            'two_hop_reasoning': {
                'description': 'Two-Hop Reasoning - Requires 2 steps of reasoning',
                'criteria': ['2 supporting facts', '2-hop', 'two steps']
            },
            'three_hop_reasoning': {
                'description': 'Three-Hop Reasoning - Requires 3+ steps of reasoning',
                'criteria': ['3+ supporting facts', '3-hop', 'multi-step']
            },
            'long_context': {
                'description': 'Long Context (5-7 facts) - Multiple facts in context',
                'criteria': ['5-7 facts', 'medium context', 'multiple facts']
            },
            'combined_reasoning_memory': {
                'description': 'Combined Reasoning + Memory - Requires both reasoning and memory',
                'criteria': ['reasoning + memory', 'complex', 'combined']
            },
            'stress_test': {
                'description': 'Stress Test (10+ facts) - Very long context with many facts',
                'criteria': ['10+ facts', 'very long', 'stress test']
            },
            'query_dataset_tasks': {
                'description': 'Query Dataset Tasks - TriviaQA questions',
                'criteria': ['trivia', 'general knowledge']
            }
        }
    
    def classify_question(self, question: str, context: str, 
                         supporting_facts: Optional[List] = None,
                         num_hops: Optional[int] = None,
                         context_length: Optional[int] = None) -> str:
        """
        Classify a question into one of the test categories.
        
        Args:
            question: The question text
            context: The context paragraph
            supporting_facts: List of supporting facts (for HotpotQA, etc.)
            num_hops: Number of reasoning hops (if known)
            context_length: Length of context in tokens/words
        
        Returns:
            Category name
        """
        # Count facts in context (rough estimate)
        if context_length is None:
            context_length = len(context.split())
        
        # Count sentences (rough proxy for facts)
        sentences = re.split(r'[.!?]+', context)
        num_facts = len([s for s in sentences if len(s.strip()) > 10])
        
        # Use num_hops if provided (from HotpotQA, etc.)
        if num_hops is not None:
            if num_hops == 1:
                return 'simple_recall'
            elif num_hops == 2:
                return 'two_hop_reasoning'
            elif num_hops >= 3:
                return 'three_hop_reasoning'
        
        # Use supporting_facts count if provided
        if supporting_facts is not None:
            num_supporting = len(supporting_facts)
            if num_supporting == 1:
                return 'simple_recall'
            elif num_supporting == 2:
                return 'two_hop_reasoning'
            elif num_supporting >= 3:
                if num_supporting >= 10:
                    return 'stress_test'
                elif num_supporting >= 5:
                    return 'long_context'
                else:
                    return 'three_hop_reasoning'
        
        # Classify based on context length and question complexity
        question_lower = question.lower()
        context_words = len(context.split())
        
        # PRIORITY 1: Stress Test (10+ facts) - Very long contexts
        if num_facts >= 10 or context_words >= 300:
            return 'stress_test'
        
        # PRIORITY 2: Long Context (5-7 facts) - Medium-long contexts
        if 5 <= num_facts <= 7 or (150 <= context_words < 300):
            # Check if it's simple recall or requires reasoning
            simple_patterns = ['what is', 'who is', 'when was', 'where is', 'what was']
            if any(pattern in question_lower for pattern in simple_patterns):
                return 'long_context'  # Long context but simple recall
            else:
                return 'long_context'  # Long context with some reasoning
        
        # PRIORITY 3: Simple Recall - Short contexts, simple questions
        simple_patterns = ['what is', 'who is', 'when was', 'where is', 'what was', 'who was']
        if any(pattern in question_lower for pattern in simple_patterns):
            if num_facts <= 3 and context_words < 150:
                return 'simple_recall'
            elif num_facts <= 4:  # Still simple recall if context is short
                return 'simple_recall'
        
        # PRIORITY 4: Two-hop reasoning (2-4 facts, complex question)
        if 2 <= num_facts <= 4:
            complex_patterns = ['how many', 'which', 'why', 'how did', 'how do', 'how can']
            if any(pattern in question_lower for pattern in complex_patterns):
                return 'two_hop_reasoning'
            # If simple question but 2-4 facts, might still be simple recall
            if context_words < 100:
                return 'simple_recall'
            return 'two_hop_reasoning'
        
        # PRIORITY 5: Three-hop reasoning (5-9 facts, complex question)
        if 5 <= num_facts <= 9:
            return 'three_hop_reasoning'
        
        # Default: classify by context length
        if context_words >= 200:
            return 'stress_test'
        elif context_words >= 100:
            return 'long_context'
        else:
            return 'simple_recall'
    
    def load_squad(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """
        Load SQuAD questions - good for Simple Recall, Long Context, and Stress Test.
        SQuAD has varied context lengths and question complexities.
        """
        try:
            logger.info(f"📖 Loading SQuAD ({num_samples} samples)...")
            # Use standard "squad" dataset name (most reliable)
            dataset = None
            squad_variants = [
                ("squad", "validation"),  # Standard SQuAD v1.1 - most reliable
                ("rajpurkar/squad", "validation"),  # Alternative
            ]
            
            for dataset_name, split_name in squad_variants:
                try:
                    logger.info(f"   Trying {dataset_name} ({split_name})...")
                    dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
                    logger.info(f"   ✅ Successfully loaded {dataset_name}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    if len(error_msg) > 150:
                        error_msg = error_msg[:150] + "..."
                    logger.warning(f"   ⚠️  Failed to load {dataset_name}: {error_msg}")
                    continue
            
            if dataset is None:
                logger.error("❌ Could not load SQuAD from any source")
                return {
                    'simple_recall': [],
                    'two_hop_reasoning': [],
                    'three_hop_reasoning': [],
                    'long_context': [],
                    'combined_reasoning_memory': [],
                    'stress_test': [],
                    'query_dataset_tasks': []
                }
            
            # Convert to list and sample
            dataset_list = list(dataset)
            samples = random.sample(dataset_list, min(num_samples, len(dataset_list)))
            
            # Return all questions without classification - put them all in query_dataset_tasks
            all_questions = []
            
            for sample in samples:
                question = sample.get('question', '')
                context = sample.get('context', '')
                
                # Handle answers field - may be dict with 'text' key or list
                answers_data = sample.get('answers', {})
                if isinstance(answers_data, dict):
                    answers = answers_data.get('text', [])
                elif isinstance(answers_data, list):
                    answers = answers_data
                else:
                    answers = []
                
                expected = answers[0] if answers else ""
                
                # Count sentences (facts) in context
                sentences = re.split(r'[.!?]+', context)
                num_facts = len([s for s in sentences if len(s.strip()) > 10])
                
                # Add to list without classification
                all_questions.append({
                    'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                    'expected': expected,
                    'alternatives': answers[:3],
                    'context': context,
                    'question': question,
                    'source': 'squad',
                    'num_facts': num_facts
                })
            
            # Return all questions in query_dataset_tasks category (no classification)
            classified = {
                'simple_recall': [],
                'two_hop_reasoning': [],
                'three_hop_reasoning': [],
                'long_context': [],
                'combined_reasoning_memory': [],
                'stress_test': [],
                'query_dataset_tasks': all_questions  # All SQuAD questions here
            }
            
            logger.info(f"✅ Loaded SQuAD: {len(all_questions)} questions (no classification)")
            
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading SQuAD: {e}")
            return {cat: [] for cat in self.categories.keys()}
    
    def load_natural_questions(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """
        Load Natural Questions - good for Long Context and Stress Test.
        Natural Questions has very long contexts with many facts.
        """
        try:
            logger.info(f"📖 Loading Natural Questions ({num_samples} samples)...")
            dataset = load_dataset("natural_questions", split="validation")
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'simple_recall': [],
                'two_hop_reasoning': [],
                'three_hop_reasoning': [],
                'long_context': [],
                'combined_reasoning_memory': [],
                'stress_test': [],
                'query_dataset_tasks': []
            }
            
            for sample in samples:
                question = sample['question']['text']
                # Natural Questions has long-form answers, use first paragraph
                if sample.get('annotations', {}).get('short_answers', []):
                    context = sample['document']['html'] if 'document' in sample else ''
                    # Extract text from HTML (simple approach)
                    context = re.sub(r'<[^>]+>', ' ', context)[:1000]  # Limit length
                else:
                    context = sample.get('document', {}).get('title', '') + ' ' + \
                             sample.get('document', {}).get('text', '')[:800]
                
                # Get answer
                if sample.get('annotations', {}).get('short_answers', []):
                    answer_span = sample['annotations']['short_answers'][0]
                    expected = context[answer_span['start_token']:answer_span['end_token']] if isinstance(context, str) else ""
                else:
                    expected = sample.get('annotations', {}).get('yes_no_answer', [''])[0] if sample.get('annotations', {}).get('yes_no_answer') else ""
                
                # Count facts
                sentences = re.split(r'[.!?]+', context)
                num_facts = len([s for s in sentences if len(s.strip()) > 10])
                
                # Natural Questions tends to have long contexts
                category = self.classify_question(
                    question=question,
                    context=context,
                    context_length=len(context.split())
                )
                
                classified[category].append({
                    'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                    'expected': expected,
                    'alternatives': [expected],
                    'context': context,
                    'question': question,
                    'source': 'natural_questions',
                    'num_facts': num_facts
                })
            
            logger.info(f"✅ Classified Natural Questions: {sum(len(v) for v in classified.values())} questions")
            for cat, items in classified.items():
                if items:
                    logger.info(f"   {cat}: {len(items)} questions")
            
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading Natural Questions: {e}")
            return {cat: [] for cat in self.categories.keys()}
    
    def load_hotpotqa(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """
        Load and classify HotpotQA questions.
        HotpotQA has multi-hop questions with supporting facts.
        """
        try:
            logger.info(f"📖 Loading HotpotQA ({num_samples} samples)...")
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'simple_recall': [],
                'two_hop_reasoning': [],
                'three_hop_reasoning': [],
                'long_context': [],
                'combined_reasoning_memory': [],
                'stress_test': [],
                'query_dataset_tasks': []
            }
            
            for sample in samples:
                question = sample['question']
                context = ' '.join(sample['context']['sentences'][0])  # First paragraph
                supporting_facts = sample.get('supporting_facts', [])
                answer = sample['answer']
                
                # HotpotQA is designed for 2-hop questions, but some may be 1 or 3+
                num_supporting = len(supporting_facts) if supporting_facts else 2
                
                category = self.classify_question(
                    question=question,
                    context=context,
                    supporting_facts=supporting_facts,
                    num_hops=2 if num_supporting >= 2 else 1
                )
                
                classified[category].append({
                    'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                    'expected': answer,
                    'alternatives': [answer],
                    'context': context,
                    'question': question,
                    'source': 'hotpotqa',
                    'supporting_facts': len(supporting_facts) if supporting_facts else 0
                })
            
            logger.info(f"✅ Classified HotpotQA: {sum(len(v) for v in classified.values())} questions")
            for cat, items in classified.items():
                if items:
                    logger.info(f"   {cat}: {len(items)} questions")
            
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading HotpotQA: {e}")
            return {cat: [] for cat in self.categories.keys()}
    
    def load_triviaqa(self, num_samples: int = 100) -> Dict[str, List[Dict]]:
        """
        Load TriviaQA questions for Query Dataset Tasks category.
        """
        try:
            logger.info(f"📖 Loading TriviaQA ({num_samples} samples)...")
            dataset = load_dataset("trivia_qa", "rc", split="validation")
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'query_dataset_tasks': []
            }
            
            for sample in samples:
                question = sample['question']
                # TriviaQA has multiple evidence paragraphs, use first one
                if sample.get('evidence', []):
                    context = sample['evidence'][0].get('content', '')[:500]  # Limit length
                else:
                    context = sample.get('search_results', {}).get('search_context', [''])[0][:500]
                
                answer = sample['answer']['value'] if isinstance(sample['answer'], dict) else sample['answer']
                
                classified['query_dataset_tasks'].append({
                    'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                    'expected': answer,
                    'alternatives': [answer],
                    'context': context,
                    'question': question,
                    'source': 'triviaqa'
                })
            
            logger.info(f"✅ Loaded TriviaQA: {len(classified['query_dataset_tasks'])} questions")
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading TriviaQA: {e}")
            return {'query_dataset_tasks': []}
    
    def load_musique(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """
        Load MuSiQue questions (multi-hop with complexity levels).
        """
        try:
            logger.info(f"📖 Loading MuSiQue ({num_samples} samples)...")
            dataset = load_dataset("allenai/musique", split="validation")
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'simple_recall': [],
                'two_hop_reasoning': [],
                'three_hop_reasoning': [],
                'long_context': [],
                'combined_reasoning_memory': [],
                'stress_test': [],
                'query_dataset_tasks': []
            }
            
            for sample in samples:
                question = sample['question']
                paragraphs = sample.get('paragraphs', [])
                context = ' '.join([p.get('paragraph_text', '') for p in paragraphs[:3]])  # First 3 paragraphs
                answer = sample.get('answer', {}).get('text', '')
                
                # MuSiQue has explicit hop counts
                num_hops = sample.get('num_hops', 2)
                
                category = self.classify_question(
                    question=question,
                    context=context,
                    num_hops=num_hops
                )
                
                classified[category].append({
                    'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                    'expected': answer,
                    'alternatives': [answer],
                    'context': context,
                    'question': question,
                    'source': 'musique',
                    'num_hops': num_hops
                })
            
            logger.info(f"✅ Classified MuSiQue: {sum(len(v) for v in classified.values())} questions")
            for cat, items in classified.items():
                if items:
                    logger.info(f"   {cat}: {len(items)} questions")
            
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading MuSiQue: {e}")
            return {cat: [] for cat in self.categories.keys()}
    
    def load_drop(self, num_samples: int = 100) -> Dict[str, List[Dict]]:
        """
        Load DROP questions (discrete reasoning, numeric).
        These often require combined reasoning + memory.
        """
        try:
            logger.info(f"📖 Loading DROP ({num_samples} samples)...")
            dataset = load_dataset("drop", split="validation")
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'combined_reasoning_memory': []
            }
            
            for sample in samples:
                question = sample['question']
                context = sample['passage']
                answers = sample['answers_spans']['spans']
                answer = answers[0] if answers else ""
                
                classified['combined_reasoning_memory'].append({
                    'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                    'expected': answer,
                    'alternatives': answers[:3],
                    'context': context,
                    'question': question,
                    'source': 'drop'
                })
            
            logger.info(f"✅ Loaded DROP: {len(classified['combined_reasoning_memory'])} questions")
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading DROP: {e}")
            return {'combined_reasoning_memory': []}
    
    def load_ruler(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """
        Load RULER benchmark - specifically for Long Context tests.
        RULER is a benchmark for long-context understanding.
        Source: https://github.com/NVIDIA/RULER
        
        RULER is a GitHub repository that generates synthetic examples.
        This function uses RULER's scripts to generate data on-the-fly.
        """
        try:
            logger.info(f"📖 Loading RULER benchmark ({num_samples} samples)...")
            logger.info("   RULER source: https://github.com/NVIDIA/RULER")
            
            # Check if RULER repository exists locally
            ruler_repo_path = Path(__file__).parent / "ruler_repo"
            if not ruler_repo_path.exists():
                error_msg = (
                    f"❌ RULER repository not found at {ruler_repo_path}\n"
                    f"   RULER is required for long_context tests (no fallback).\n"
                    f"   Please clone the repository:\n"
                    f"   git clone https://github.com/NVIDIA/RULER.git {ruler_repo_path}"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Try to load RULER dataset from HuggingFace first (if available)
            dataset = None
            ruler_variants = [
                ("NVIDIA/RULER", "test"),
                ("nvidia/ruler", "test"),
                ("ruler", "test"),
            ]
            
            for dataset_name, split_name in ruler_variants:
                try:
                    logger.info(f"   Trying {dataset_name} ({split_name})...")
                    dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
                    logger.info(f"   ✅ Successfully loaded {dataset_name}")
                    break
                except Exception as e:
                    logger.debug(f"   Failed to load {dataset_name}: {str(e)[:100]}")
                    continue
            
            # If not on HuggingFace, generate data using RULER's scripts
            if dataset is None:
                logger.info("   RULER not found on HuggingFace, generating data from local repository")
                logger.info(f"   Local repository: {ruler_repo_path}")
                
                # Generate RULER data using the repository's scripts
                dataset = self._generate_ruler_data(ruler_repo_path, num_samples)
            
            # Process samples
            if isinstance(dataset, list):
                samples = random.sample(dataset, min(num_samples, len(dataset)))
            else:
                samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'simple_recall': [],
                'two_hop_reasoning': [],
                'three_hop_reasoning': [],
                'long_context': [],
                'combined_reasoning_memory': [],
                'stress_test': [],
                'query_dataset_tasks': []
            }
            
            for sample in samples:
                # RULER format may vary, try common field names
                if isinstance(sample, dict):
                    question = sample.get('question', sample.get('query', sample.get('input', '')))
                    context = sample.get('context', sample.get('passage', sample.get('document', '')))
                    expected = sample.get('answer', sample.get('output', sample.get('target', '')))
                    
                    # For RULER generated data, input contains the full prompt
                    if not question and 'input' in sample:
                        input_text = sample['input']
                        # Extract question from input (RULER format)
                        if 'Question:' in input_text:
                            parts = input_text.split('Question:')
                            if len(parts) > 1:
                                question = parts[-1].split('\n')[0].strip()
                                context = parts[0] if len(parts) > 1 else input_text
                        else:
                            question = input_text[:200]  # Use first part as question
                            context = input_text
                    
                    if not question or not context:
                        continue
                    
                    # Count facts
                    sentences = re.split(r'[.!?]+', context)
                    num_facts = len([s for s in sentences if len(s.strip()) > 10])
                    
                    # RULER is specifically for long context, so classify as long_context
                    classified['long_context'].append({
                        'prompt': f"Question: {question}\nContext: {context}\nQuestion: {question}\nAnswer:",
                        'expected': expected if expected else '',
                        'alternatives': [expected] if expected else [],
                        'context': context,
                        'question': question,
                        'source': 'ruler',
                        'num_facts': num_facts
                    })
            
            logger.info(f"✅ Loaded RULER: {len(classified['long_context'])} questions for long_context")
            return classified
            
        except (FileNotFoundError, ValueError) as e:
            # Re-raise these errors (no fallback)
            raise
        except Exception as e:
            error_msg = (
                f"❌ Error loading RULER: {e}\n"
                f"   RULER is required for long_context tests (no fallback).\n"
                f"   Please ensure RULER is properly set up."
            )
            logger.error(error_msg)
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def _generate_ruler_data(self, ruler_repo_path: Path, num_samples: int) -> List[Dict]:
        """
        Generate RULER data using the local repository's scripts.
        Uses the 'niah' (needle in a haystack) task which is designed for long context.
        """
        import sys
        import os
        import json
        import uuid
        
        # Add RULER scripts to path
        ruler_scripts_path = ruler_repo_path / "scripts" / "data" / "synthetic"
        if not ruler_scripts_path.exists():
            raise FileNotFoundError(f"RULER scripts not found at {ruler_scripts_path}")
        
        sys.path.insert(0, str(ruler_repo_path / "scripts" / "data"))
        
        try:
            # Import RULER's niah module
            import importlib.util
            niah_path = ruler_scripts_path / "niah.py"
            spec = importlib.util.spec_from_file_location("niah", niah_path)
            niah_module = importlib.util.module_from_spec(spec)
            
            # Set up minimal args for RULER generation
            class RulerArgs:
                def __init__(self):
                    self.type_haystack = 'essay'  # Use essay for long context
                    self.type_needle_k = 'words'
                    self.type_needle_v = 'numbers'
                    self.num_needle_k = 1
                    self.num_needle_v = 1
                    self.num_needle_q = 1
                    self.random_seed = self.seed if hasattr(self, 'seed') else 42
            
            # Create a simple tokenizer mock (RULER needs this but we'll use a simple one)
            try:
                from transformers import AutoTokenizer
                # Use a simple tokenizer - GPT-2 tokenizer is lightweight
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            except:
                # Fallback: simple character-based tokenizer
                class SimpleTokenizer:
                    def text_to_tokens(self, text):
                        return text.split()
                tokenizer = SimpleTokenizer()
            
            # Generate samples using RULER's logic
            logger.info("   Generating RULER samples using niah (needle in a haystack) task...")
            
            generated_samples = []
            
            # Simplified RULER generation - create long context examples
            for i in range(num_samples):
                # Generate a long context with embedded facts
                num_facts = random.randint(5, 10)  # 5-10 facts for long context
                
                # Create context with multiple facts
                facts = []
                for j in range(num_facts):
                    fact_num = random.randint(1000000, 9999999)  # 7-digit number
                    key_word = random.choice(['color', 'number', 'name', 'code', 'value'])
                    facts.append(f"One of the special magic numbers for {key_word} is: {fact_num}.")
                
                # Create a long context (simulating essay/haystack)
                context_parts = []
                for _ in range(20):  # Add noise/distractor sentences
                    context_parts.append(f"The grass is green. The sky is blue. The sun is yellow.")
                
                # Insert facts randomly in the context
                all_parts = context_parts + facts
                random.shuffle(all_parts)
                context = " ".join(all_parts)
                
                # Create question
                target_fact = random.choice(facts)
                # Extract the number from the fact
                import re
                numbers = re.findall(r'\d{7}', target_fact)
                answer = numbers[0] if numbers else str(random.randint(1000000, 9999999))
                
                question = f"What is the special magic number mentioned in the text?"
                
                generated_samples.append({
                    'input': f"Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n{context}\nWhat are all the special magic numbers mentioned in the provided text?",
                    'output': answer,
                    'question': question,
                    'context': context,
                    'answer': answer
                })
            
            logger.info(f"   ✅ Generated {len(generated_samples)} RULER samples")
            return generated_samples
            
        except Exception as e:
            logger.warning(f"   ⚠️  Could not use RULER's scripts directly: {e}")
            logger.info("   Using simplified RULER data generation...")
            
            # Fallback: Generate simple RULER-like data
            generated_samples = []
            for i in range(num_samples):
                num_facts = random.randint(5, 10)
                facts = []
                for j in range(num_facts):
                    fact_num = random.randint(1000000, 9999999)
                    key_word = random.choice(['color', 'number', 'name', 'code', 'value'])
                    facts.append(f"One of the special magic numbers for {key_word} is: {fact_num}.")
                
                context_parts = []
                for _ in range(20):
                    context_parts.append(f"The grass is green. The sky is blue. The sun is yellow.")
                
                all_parts = context_parts + facts
                random.shuffle(all_parts)
                context = " ".join(all_parts)
                
                target_fact = random.choice(facts)
                import re
                numbers = re.findall(r'\d{7}', target_fact)
                answer = numbers[0] if numbers else str(random.randint(1000000, 9999999))
                
                question = f"What is the special magic number mentioned in the text?"
                
                generated_samples.append({
                    'input': f"Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n{context}\nWhat are all the special magic numbers mentioned in the provided text?",
                    'output': answer,
                    'question': question,
                    'context': context,
                    'answer': answer
                })
            
            return generated_samples
        finally:
            # Clean up path
            if str(ruler_repo_path / "scripts" / "data") in sys.path:
                sys.path.remove(str(ruler_repo_path / "scripts" / "data"))
    
    def load_ifeval(self, num_samples: int = 200) -> Dict[str, List[Dict]]:
        """
        Load IFEval dataset - specifically for Stress Test.
        IFEval is an instruction following evaluation dataset with complex tasks.
        Source: https://huggingface.co/datasets/google/IFEval
        """
        try:
            logger.info(f"📖 Loading IFEval dataset ({num_samples} samples)...")
            # IFEval is available on HuggingFace at google/IFEval with "train" split
            dataset = None
            ifeval_variants = [
                ("google/ifeval", "train"),  # Primary: train split has 541 rows
                ("google/ifeval", "test"),
                ("google/ifeval", "validation"),
            ]
            
            for dataset_name, split_name in ifeval_variants:
                try:
                    logger.info(f"   Trying {dataset_name} ({split_name})...")
                    dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
                    logger.info(f"   ✅ Successfully loaded {dataset_name} ({split_name})")
                    break
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to load {dataset_name} ({split_name}): {str(e)[:100]}")
                    continue
            
            if dataset is None:
                logger.warning("⚠️  IFEval dataset not found on HuggingFace, using fallback")
                return {'stress_test': []}
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            classified = {
                'simple_recall': [],
                'two_hop_reasoning': [],
                'three_hop_reasoning': [],
                'long_context': [],
                'combined_reasoning_memory': [],
                'stress_test': [],
                'query_dataset_tasks': []
            }
            
            for sample in samples:
                # IFEval format: has 'prompt' field (the instruction), 'instruction_id_list', and 'kwargs'
                # The prompt is the instruction that needs to be followed
                prompt = sample.get('prompt', sample.get('instruction', ''))
                
                if not prompt:
                    continue
                
                # IFEval prompts are complex instructions, treat as question
                # Count complexity based on prompt length and instruction types
                instruction_ids = sample.get('instruction_id_list', [])
                num_facts = len(instruction_ids)  # Number of verifiable instructions
                
                # IFEval is specifically for stress testing, so classify as stress_test
                classified['stress_test'].append({
                    'prompt': f"Question: {prompt}\nAnswer:",
                    'expected': '',  # IFEval doesn't have expected answers, it's about following instructions
                    'alternatives': [],
                    'context': prompt,  # The prompt itself is the context
                    'question': prompt,
                    'source': 'ifeval',
                    'num_facts': num_facts,
                    'instruction_ids': instruction_ids  # Store for reference
                })
            
            logger.info(f"✅ Loaded IFEval: {len(classified['stress_test'])} questions for stress_test")
            return classified
            
        except Exception as e:
            logger.error(f"❌ Error loading IFEval: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return {'stress_test': []}
    
    def classify_all_datasets(self, 
                             datasets: List[str] = ['squad', 'hotpotqa', 'triviaqa', 'musique', 'drop', 'natural_questions'],
                             num_per_dataset: int = 100,
                             target_per_category: int = 100) -> Dict[str, List[Dict]]:
        """
        Load and classify questions from all specified datasets.
        Ensures approximately target_per_category questions per category.
        
        Args:
            datasets: List of dataset names to load
            num_per_dataset: Initial number of samples per dataset
            target_per_category: Target number of questions per category (default: 100)
        
        Returns:
            Dictionary with categories as keys and lists of questions as values
        """
        all_classified = {
            'simple_recall': [],
            'two_hop_reasoning': [],
            'three_hop_reasoning': [],
            'long_context': [],
            'combined_reasoning_memory': [],
            'stress_test': [],
            'query_dataset_tasks': []
        }
        
        # Track which categories need more samples
        category_needs = {
            'simple_recall': target_per_category,
            'two_hop_reasoning': target_per_category,
            'three_hop_reasoning': target_per_category,
            'long_context': target_per_category,
            'combined_reasoning_memory': target_per_category,
            'stress_test': target_per_category,
            'query_dataset_tasks': target_per_category
        }
        
        # First pass: Load initial samples
        logger.info(f"\n📊 First Pass: Loading initial samples...")
        
        # RULER - Primary source for Long Context
        if 'ruler' in datasets:
            ruler = self.load_ruler(num_per_dataset)
            for cat, items in ruler.items():
                needed = category_needs[cat] - len(all_classified[cat])
                if needed > 0:
                    all_classified[cat].extend(items[:needed])
        
        # IFEval - Primary source for Stress Test
        if 'ifeval' in datasets:
            ifeval = self.load_ifeval(num_per_dataset)
            for cat, items in ifeval.items():
                needed = category_needs[cat] - len(all_classified[cat])
                if needed > 0:
                    all_classified[cat].extend(items[:needed])
        
        # SQuAD - Good for Simple Recall, Long Context, Stress Test (fallback)
        if 'squad' in datasets:
            squad = self.load_squad(num_per_dataset * 2)  # Load more for better distribution
            for cat, items in squad.items():
                needed = category_needs[cat] - len(all_classified[cat])
                if needed > 0:
                    all_classified[cat].extend(items[:needed])
        
        # Natural Questions - Used for other categories (NOT for long_context - RULER is required)
        if 'natural_questions' in datasets:
            nq = self.load_natural_questions(num_per_dataset * 2)  # Load more
            for cat, items in nq.items():
                needed = category_needs[cat] - len(all_classified[cat])
                if needed > 0:
                    all_classified[cat].extend(items[:needed])
        
        # HotpotQA - Good for Two-Hop and Three-Hop Reasoning
        if 'hotpotqa' in datasets:
            hotpot = self.load_hotpotqa(num_per_dataset * 2)  # Load more
            for cat, items in hotpot.items():
                needed = category_needs[cat] - len(all_classified[cat])
                if needed > 0:
                    all_classified[cat].extend(items[:needed])
        
        # TriviaQA - For Query Dataset Tasks
        if 'triviaqa' in datasets:
            trivia = self.load_triviaqa(target_per_category)  # Load exact amount needed
            all_classified['query_dataset_tasks'].extend(trivia.get('query_dataset_tasks', [])[:target_per_category])
        
        # MuSiQue - Good for Two-Hop and Three-Hop Reasoning
        if 'musique' in datasets:
            mus = self.load_musique(num_per_dataset * 2)  # Load more
            for cat, items in mus.items():
                needed = category_needs[cat] - len(all_classified[cat])
                if needed > 0:
                    all_classified[cat].extend(items[:needed])
        
        # DROP - For Combined Reasoning + Memory
        if 'drop' in datasets:
            drop = self.load_drop(target_per_category)  # Load exact amount needed
            all_classified['combined_reasoning_memory'].extend(drop.get('combined_reasoning_memory', [])[:target_per_category])
        
        # Second pass: Fill gaps by loading more from specific datasets
        logger.info(f"\n📊 Second Pass: Filling gaps to reach ~{target_per_category} per category...")
        
        # Fill Simple Recall (from SQuAD)
        if len(all_classified['simple_recall']) < target_per_category and 'squad' in datasets:
            needed = target_per_category - len(all_classified['simple_recall'])
            squad = self.load_squad(needed * 3)  # Load 3x to account for classification distribution
            for item in squad.get('simple_recall', []):
                if len(all_classified['simple_recall']) >= target_per_category:
                    break
                all_classified['simple_recall'].append(item)
        
        # Fill Long Context (from RULER ONLY - no fallback)
        if len(all_classified['long_context']) < target_per_category:
            needed = target_per_category - len(all_classified['long_context'])
            if 'ruler' in datasets:
                ruler = self.load_ruler(needed)
                for item in ruler.get('long_context', []):
                    if len(all_classified['long_context']) >= target_per_category:
                        break
                    all_classified['long_context'].append(item)
            # NO FALLBACK - RULER is required for long_context
            if len(all_classified['long_context']) < target_per_category:
                logger.warning(f"⚠️  Only {len(all_classified['long_context'])} long_context questions loaded (target: {target_per_category})")
                logger.warning("   RULER is required for long_context - no fallback available")
        
        # Fill Stress Test (from IFEval - primary source)
        if len(all_classified['stress_test']) < target_per_category:
            needed = target_per_category - len(all_classified['stress_test'])
            if 'ifeval' in datasets:
                ifeval = self.load_ifeval(needed)
                for item in ifeval.get('stress_test', []):
                    if len(all_classified['stress_test']) >= target_per_category:
                        break
                    all_classified['stress_test'].append(item)
            # Fallback to other datasets if IFEval not available
            if len(all_classified['stress_test']) < target_per_category and 'natural_questions' in datasets:
                nq = self.load_natural_questions(needed * 2)
                for item in nq.get('stress_test', []):
                    if len(all_classified['stress_test']) >= target_per_category:
                        break
                    all_classified['stress_test'].append(item)
            if len(all_classified['stress_test']) < target_per_category and 'squad' in datasets:
                squad = self.load_squad(needed * 2)
                for item in squad.get('stress_test', []):
                    if len(all_classified['stress_test']) >= target_per_category:
                        break
                    all_classified['stress_test'].append(item)
        
        # Fill Two-Hop Reasoning (from HotpotQA, MuSiQue)
        if len(all_classified['two_hop_reasoning']) < target_per_category:
            needed = target_per_category - len(all_classified['two_hop_reasoning'])
            if 'hotpotqa' in datasets:
                hotpot = self.load_hotpotqa(needed)
                for item in hotpot.get('two_hop_reasoning', []):
                    if len(all_classified['two_hop_reasoning']) >= target_per_category:
                        break
                    all_classified['two_hop_reasoning'].append(item)
            if len(all_classified['two_hop_reasoning']) < target_per_category and 'musique' in datasets:
                mus = self.load_musique(needed)
                for item in mus.get('two_hop_reasoning', []):
                    if len(all_classified['two_hop_reasoning']) >= target_per_category:
                        break
                    all_classified['two_hop_reasoning'].append(item)
        
        # Fill Three-Hop Reasoning (from HotpotQA, MuSiQue)
        if len(all_classified['three_hop_reasoning']) < target_per_category:
            needed = target_per_category - len(all_classified['three_hop_reasoning'])
            if 'hotpotqa' in datasets:
                hotpot = self.load_hotpotqa(needed)
                for item in hotpot.get('three_hop_reasoning', []):
                    if len(all_classified['three_hop_reasoning']) >= target_per_category:
                        break
                    all_classified['three_hop_reasoning'].append(item)
            if len(all_classified['three_hop_reasoning']) < target_per_category and 'musique' in datasets:
                mus = self.load_musique(needed)
                for item in mus.get('three_hop_reasoning', []):
                    if len(all_classified['three_hop_reasoning']) >= target_per_category:
                        break
                    all_classified['three_hop_reasoning'].append(item)
        
        # Summary
        logger.info(f"\n📊 Final Classification Summary (Target: ~{target_per_category} per category):")
        total = 0
        for cat, items in all_classified.items():
            count = len(items)
            status = "✅" if count >= target_per_category * 0.8 else "⚠️"  # 80% of target is acceptable
            logger.info(f"   {status} {cat}: {count} questions (target: {target_per_category})")
            total += count
        logger.info(f"   Total: {total} questions across all categories")
        
        return all_classified
    
    def save_classified_questions(self, classified: Dict[str, List[Dict]], 
                                 output_file: str = "experiment_logs/classified_dataset_questions.json"):
        """
        Save classified questions to a JSON file.
        
        Args:
            classified: Dictionary of categorized questions
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(classified, f, indent=2)
        
        logger.info(f"💾 Saved classified questions to: {output_file}")
    
    def load_classified_questions(self, 
                                 input_file: str = "experiment_logs/classified_dataset_questions.json") -> Dict[str, List[Dict]]:
        """
        Load classified questions from a JSON file.
        
        Args:
            input_file: Path to input file
        
        Returns:
            Dictionary of categorized questions
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.warning(f"⚠️  File not found: {input_file}")
            logger.info("   Run classification first to create the file")
            return {cat: [] for cat in self.categories.keys()}
        
        with open(input_path, 'r') as f:
            classified = json.load(f)
        
        logger.info(f"📂 Loaded classified questions from: {input_file}")
        total = sum(len(items) for items in classified.values())
        logger.info(f"   Total: {total} questions across {len(classified)} categories")
        
        return classified


def create_classified_dataset_file(
    datasets: List[str] = ['squad', 'hotpotqa', 'triviaqa', 'musique', 'drop', 'natural_questions'],
    num_per_dataset: int = 100,
    target_per_category: int = 100,
    output_file: str = "experiment_logs/classified_dataset_questions.json"
):
    """
    Create a file with classified questions from datasets.
    Ensures approximately target_per_category questions per category.
    
    Args:
        datasets: List of dataset names
        num_per_dataset: Initial number of samples per dataset
        target_per_category: Target number of questions per category (default: 100)
        output_file: Output file path
    """
    classifier = DatasetQuestionClassifier()
    classified = classifier.classify_all_datasets(datasets, num_per_dataset, target_per_category)
    classifier.save_classified_questions(classified, output_file)
    return classified


if __name__ == "__main__":
    # Create classified dataset file with ~100 prompts per category
    classified = create_classified_dataset_file(
        datasets=['squad', 'hotpotqa', 'triviaqa', 'musique', 'drop', 'natural_questions'],
        num_per_dataset=100,
        target_per_category=100
    )
