"""
Task and dataset interfaces for multi-agent evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import random
from pathlib import Path


class TaskType(Enum):
    """Types of tasks for evaluation."""
    MATH_REASONING = "math_reasoning"
    MULTI_HOP_QA = "multi_hop_qa"
    CODE_GENERATION = "code_generation"
    COMMONSENSE_QA = "commonsense_qa"
    INSTRUCTION_FOLLOWING = "instruction_following"


class DatasetType(Enum):
    """Types of datasets for evaluation."""
    GSM8K = "gsm8k"
    MATH = "math"
    HOTPOTQA = "hotpotqa"
    HUMANEVAL = "humaneval"
    HELLASWAG = "hellaswag"
    ALPACAEVAL = "alpacaeval"


@dataclass
class TaskItem:
    """Individual task item for evaluation."""
    item_id: str
    task_type: TaskType
    dataset: DatasetType
    question: str
    ground_truth: Union[str, List[str]]
    metadata: Dict[str, Any]
    
    # Evaluation-specific fields
    difficulty_level: Optional[str] = None
    domain: Optional[str] = None
    expected_reasoning_steps: Optional[int] = None


@dataclass
class TaskConfig:
    """Configuration for task evaluation."""
    task_type: TaskType
    dataset: DatasetType
    num_samples: int
    difficulty_filter: Optional[List[str]] = None
    domain_filter: Optional[List[str]] = None
    random_seed: int = 42
    max_tokens: int = 2000
    timeout_seconds: float = 300.0


class Task(ABC):
    """
    Abstract base class for tasks in multi-agent evaluation.
    
    Each task type implements specific evaluation logic and metrics.
    """
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.task_items: List[TaskItem] = []
        self.load_dataset()
    
    @abstractmethod
    def load_dataset(self):
        """Load dataset for the task."""
        pass
    
    @abstractmethod
    def evaluate_response(
        self, 
        response: str, 
        ground_truth: Union[str, List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response against ground truth.
        
        Args:
            response: The generated response
            ground_truth: The correct answer(s)
            metadata: Additional metadata for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def get_sample_items(self, num_samples: Optional[int] = None) -> List[TaskItem]:
        """Get a sample of task items."""
        num_samples = num_samples or self.config.num_samples
        return random.sample(self.task_items, min(num_samples, len(self.task_items)))
    
    def get_item_by_id(self, item_id: str) -> Optional[TaskItem]:
        """Get a specific task item by ID."""
        for item in self.task_items:
            if item.item_id == item_id:
                return item
        return None


class MathReasoningTask(Task):
    """
    Math reasoning task implementation.
    
    Handles GSM8K and MATH datasets for mathematical problem solving.
    """
    
    def load_dataset(self):
        """Load math reasoning dataset."""
        # TODO: Implement actual dataset loading
        # This would load from HuggingFace datasets or local files
        
        if self.config.dataset == DatasetType.GSM8K:
            self._load_gsm8k()
        elif self.config.dataset == DatasetType.MATH:
            self._load_math()
        else:
            raise ValueError(f"Unsupported dataset for math reasoning: {self.config.dataset}")
    
    def _load_gsm8k(self):
        """Load GSM8K dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            # Load GSM8K test set
            dataset = load_dataset("gsm8k", "main", split="test")
            
            self.task_items = []
            for i, example in enumerate(dataset):
                item = TaskItem(
                    item_id=f"gsm8k_{i}",
                    task_type=TaskType.MATH_REASONING,
                    dataset=DatasetType.GSM8K,
                    question=example["question"],
                    ground_truth=example["answer"].strip(),  # Extract answer from "answer: X"
                    metadata={"original_answer": example["answer"]},
                    difficulty_level="easy",
                    domain="arithmetic",
                    expected_reasoning_steps=4
                )
                # Extract numeric answer from ground truth
                import re
                answer_match = re.search(r'####\s*(-?\d+\.?\d*)', example["answer"])
                if answer_match:
                    item.ground_truth = answer_match.group(1)
                self.task_items.append(item)
            
            print(f"Loaded {len(self.task_items)} GSM8K examples")
            
        except Exception as e:
            print(f"Error loading GSM8K dataset: {e}")
            print("Falling back to placeholder data")
            self.task_items = [
                TaskItem(
                    item_id="gsm8k_001",
                    task_type=TaskType.MATH_REASONING,
                    dataset=DatasetType.GSM8K,
                    question="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                    ground_truth="18",
                    metadata={"difficulty": "easy"},
                    difficulty_level="easy",
                    domain="arithmetic",
                    expected_reasoning_steps=4
                )
            ]
    
    def _load_math(self):
        """Load MATH dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            # Load MATH test set
            dataset = load_dataset("hendrycks/competition_math", split="test")
            
            self.task_items = []
            for i, example in enumerate(dataset):
                item = TaskItem(
                    item_id=f"math_{i}",
                    task_type=TaskType.MATH_REASONING,
                    dataset=DatasetType.MATH,
                    question=example["problem"],
                    ground_truth=example["solution"].split("\\boxed{")[1].split("}")[0] if "\\boxed{" in example["solution"] else example["solution"][:50],
                    metadata={
                        "level": example.get("level", "unknown"),
                        "type": example.get("type", "unknown"),
                        "solution": example["solution"]
                    },
                    difficulty_level=example.get("level", "medium").lower(),
                    domain=example.get("type", "algebra").lower(),
                    expected_reasoning_steps=5
                )
                self.task_items.append(item)
            
            print(f"Loaded {len(self.task_items)} MATH examples")
            
        except Exception as e:
            print(f"Error loading MATH dataset: {e}")
            print("Falling back to placeholder data")
            self.task_items = [
                TaskItem(
                    item_id="math_001",
                    task_type=TaskType.MATH_REASONING,
                    dataset=DatasetType.MATH,
                    question="Find all real numbers $x$ such that $x^2 + 2x + 1 = 0$.",
                    ground_truth="-1",
                    metadata={"difficulty": "medium"},
                    difficulty_level="medium",
                    domain="algebra",
                    expected_reasoning_steps=3
                )
            ]
    
    def evaluate_response(
        self, 
        response: str, 
        ground_truth: Union[str, List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate math reasoning response."""
        # Extract final answer from response
        final_answer = self._extract_final_answer(response)
        
        # Check exact match
        exact_match = self._check_exact_match(final_answer, ground_truth)
        
        # Check numerical equivalence
        numerical_match = self._check_numerical_match(final_answer, ground_truth)
        
        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(response)
        
        return {
            "exact_match": exact_match,
            "numerical_match": numerical_match,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "reasoning_steps": reasoning_steps,
            "response_length": len(response),
            "evaluation_method": "math_reasoning"
        }
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final numerical answer from response."""
        # TODO: Implement answer extraction logic
        # Look for patterns like "The answer is X" or "X" at the end
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and any(char.isdigit() for char in line):
                # Extract numbers from the line
                import re
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if numbers:
                    return numbers[-1]
        return response.strip()
    
    def _check_exact_match(self, answer: str, ground_truth: Union[str, List[str]]) -> bool:
        """Check if answer exactly matches ground truth."""
        if isinstance(ground_truth, list):
            return answer in ground_truth
        return answer == ground_truth
    
    def _check_numerical_match(self, answer: str, ground_truth: Union[str, List[str]]) -> bool:
        """Check if answer numerically matches ground truth."""
        try:
            answer_num = float(answer)
            if isinstance(ground_truth, list):
                return any(abs(answer_num - float(gt)) < 1e-6 for gt in ground_truth)
            else:
                return abs(answer_num - float(ground_truth)) < 1e-6
        except (ValueError, TypeError):
            return False
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response."""
        # TODO: Implement reasoning step extraction
        # Split by common step indicators
        steps = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and any(indicator in line.lower() for indicator in ['step', 'first', 'next', 'then', 'therefore']):
                steps.append(line)
        return steps


class MultiHopQATask(Task):
    """
    Multi-hop question answering task implementation.
    
    Handles HotpotQA dataset for complex reasoning questions.
    """
    
    def load_dataset(self):
        """Load multi-hop QA dataset."""
        if self.config.dataset == DatasetType.HOTPOTQA:
            self._load_hotpotqa()
        else:
            raise ValueError(f"Unsupported dataset for multi-hop QA: {self.config.dataset}")
    
    def _load_hotpotqa(self):
        """Load HotpotQA dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            # Load HotpotQA validation set
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            
            self.task_items = []
            for i, example in enumerate(dataset):
                item = TaskItem(
                    item_id=f"hotpotqa_{i}",
                    task_type=TaskType.MULTI_HOP_QA,
                    dataset=DatasetType.HOTPOTQA,
                    question=example["question"],
                    ground_truth=example["answer"],
                    metadata={
                        "level": example.get("level", "medium"),
                        "type": example.get("type", "comparison"),
                        "context": example.get("context", [])
                    },
                    difficulty_level=example.get("level", "medium").lower(),
                    domain="factual",
                    expected_reasoning_steps=3
                )
                self.task_items.append(item)
            
            print(f"Loaded {len(self.task_items)} HotpotQA examples")
            
        except Exception as e:
            print(f"Error loading HotpotQA dataset: {e}")
            print("Falling back to placeholder data")
            self.task_items = [
                TaskItem(
                    item_id="hotpotqa_001",
                    task_type=TaskType.MULTI_HOP_QA,
                    dataset=DatasetType.HOTPOTQA,
                    question="Which magazine was started first Arthur's Magazine or First for Women?",
                    ground_truth="Arthur's Magazine",
                    metadata={"difficulty": "medium"},
                    difficulty_level="medium",
                    domain="history",
                    expected_reasoning_steps=3
                )
            ]
    
    def evaluate_response(
        self, 
        response: str, 
        ground_truth: Union[str, List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate multi-hop QA response."""
        # Check exact match
        exact_match = self._check_exact_match(response, ground_truth)
        
        # Check F1 score for partial matches
        f1_score = self._calculate_f1_score(response, ground_truth)
        
        # Check if answer contains key entities
        entity_match = self._check_entity_match(response, ground_truth)
        
        return {
            "exact_match": exact_match,
            "f1_score": f1_score,
            "entity_match": entity_match,
            "response": response,
            "ground_truth": ground_truth,
            "evaluation_method": "multi_hop_qa"
        }
    
    def _check_exact_match(self, response: str, ground_truth: Union[str, List[str]]) -> bool:
        """Check exact match for QA."""
        response_clean = response.strip().lower()
        if isinstance(ground_truth, list):
            return any(response_clean == gt.strip().lower() for gt in ground_truth)
        return response_clean == ground_truth.strip().lower()
    
    def _calculate_f1_score(self, response: str, ground_truth: Union[str, List[str]]) -> float:
        """Calculate F1 score for response."""
        # TODO: Implement F1 calculation
        # Simple word-level F1 for now
        response_words = set(response.lower().split())
        if isinstance(ground_truth, list):
            gt_words = set(' '.join(ground_truth).lower().split())
        else:
            gt_words = set(ground_truth.lower().split())
        
        if not gt_words:
            return 0.0
        
        precision = len(response_words & gt_words) / len(response_words) if response_words else 0
        recall = len(response_words & gt_words) / len(gt_words)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _check_entity_match(self, response: str, ground_truth: Union[str, List[str]]) -> bool:
        """Check if response contains key entities from ground truth."""
        # TODO: Implement entity matching
        return True  # Placeholder


class CodeGenerationTask(Task):
    """
    Code generation task implementation.
    
    Handles HumanEval dataset for programming tasks.
    """
    
    def load_dataset(self):
        """Load code generation dataset."""
        if self.config.dataset == DatasetType.HUMANEVAL:
            self._load_humaneval()
        else:
            raise ValueError(f"Unsupported dataset for code generation: {self.config.dataset}")
    
    def _load_humaneval(self):
        """Load HumanEval dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            # Load HumanEval dataset
            dataset = load_dataset("openai/humaneval", split="test")
            
            self.task_items = []
            for i, example in enumerate(dataset):
                # Extract function signature and docstring
                prompt = example["prompt"]
                
                item = TaskItem(
                    item_id=f"humaneval_{i}",
                    task_type=TaskType.CODE_GENERATION,
                    dataset=DatasetType.HUMANEVAL,
                    question=prompt,
                    ground_truth=example["canonical_solution"],
                    metadata={
                        "task_id": example["task_id"],
                        "entry_point": example["entry_point"],
                        "test": example["test"]
                    },
                    difficulty_level="medium",
                    domain="programming",
                    expected_reasoning_steps=1
                )
                self.task_items.append(item)
            
            print(f"Loaded {len(self.task_items)} HumanEval examples")
            
        except Exception as e:
            print(f"Error loading HumanEval dataset: {e}")
            print("Falling back to placeholder data")
            self.task_items = [
                TaskItem(
                    item_id="humaneval_001",
                    task_type=TaskType.CODE_GENERATION,
                    dataset=DatasetType.HUMANEVAL,
                    question="def add_two_numbers(a, b):\n    \"\"\"\n    Add two numbers and return the result.\n    \n    Args:\n        a (int): First number\n        b (int): Second number\n    \n    Returns:\n        int: Sum of a and b\n    \"\"\"\n    pass",
                    ground_truth="return a + b",
                    metadata={"difficulty": "easy"},
                    difficulty_level="easy",
                    domain="programming",
                    expected_reasoning_steps=1
                )
            ]
    
    def evaluate_response(
        self, 
        response: str, 
        ground_truth: Union[str, List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate code generation response."""
        # Check if code compiles
        compile_success = self._check_compile_success(response)
        
        # Check if code passes tests (pass@k)
        test_results = self._run_tests(response, ground_truth)
        
        # Check code quality metrics
        quality_metrics = self._evaluate_code_quality(response)
        
        return {
            "compile_success": compile_success,
            "test_results": test_results,
            "quality_metrics": quality_metrics,
            "response": response,
            "ground_truth": ground_truth,
            "evaluation_method": "code_generation"
        }
    
    def _check_compile_success(self, code: str) -> bool:
        """Check if code compiles successfully."""
        # TODO: Implement compilation check
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _run_tests(self, code: str, ground_truth: Union[str, List[str]]) -> Dict[str, Any]:
        """Run tests on generated code."""
        # TODO: Implement test execution
        return {
            "pass_at_1": False,  # Placeholder
            "pass_at_10": False,  # Placeholder
            "pass_at_100": False,  # Placeholder
            "test_cases_passed": 0,
            "total_test_cases": 1
        }
    
    def _evaluate_code_quality(self, code: str) -> Dict[str, Any]:
        """Evaluate code quality metrics."""
        # TODO: Implement code quality evaluation
        return {
            "lines_of_code": len(code.split('\n')),
            "complexity_score": 1.0,  # Placeholder
            "readability_score": 0.8  # Placeholder
        }


class HellaSwagTask(Task):
    """
    HellaSwag commonsense reasoning task implementation.
    
    Multiple-choice task requiring commonsense understanding.
    """
    
    def load_dataset(self):
        """Load HellaSwag dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            # Load validation split
            dataset = load_dataset("Rowan/hellaswag", split="validation")
            
            # Convert to TaskItems
            self.task_items = []
            for i, example in enumerate(dataset):
                # HellaSwag format: context, endings (4 options), label
                context = example['ctx']
                endings = example['endings']
                
                # Format question with multiple choice options - clearer format
                question = f"""Context: {context}

Task: Which is the most likely continuation of the context above?

Options:
0) {endings[0]}
1) {endings[1]}
2) {endings[2]}
3) {endings[3]}

Instructions:
- Analyze each option carefully
- Consider which option makes the most sense based on commonsense reasoning
- Think about what would naturally follow in the real world
- Provide your answer as a single number (0, 1, 2, or 3)

Answer:"""
                
                item = TaskItem(
                    item_id=f"hellaswag_{i}",
                    task_type=TaskType.COMMONSENSE_QA,
                    dataset=DatasetType.HELLASWAG,
                    question=question,
                    ground_truth=str(example['label']),  # 0-3 index
                    metadata={
                        "context": context,
                        "endings": endings,
                        "activity_label": example.get('activity_label', '')
                    },
                    difficulty_level="medium",
                    domain="commonsense"
                )
                self.task_items.append(item)
            
            print(f"Loaded {len(self.task_items)} HellaSwag examples")
            
        except Exception as e:
            print(f"Error loading HellaSwag dataset: {e}")
            print("Falling back to placeholder data")
            # Placeholder data
            self.task_items = [
                TaskItem(
                    item_id="hellaswag_001",
                    task_type=TaskType.COMMONSENSE_QA,
                    dataset=DatasetType.HELLASWAG,
                    question="A man sits down at a computer. He presses the spacebar. ",
                    ground_truth="0",
                    metadata={
                        "context": "A man sits down at a computer. He presses the spacebar.",
                        "endings": [
                            "he opens a web browser.",
                            "the computer turns on.",
                            "he types a message.",
                            "the screen displays text."
                        ]
                    },
                    difficulty_level="medium",
                    domain="commonsense"
                )
            ]
    
    def evaluate_response(
        self, 
        response: str, 
        ground_truth: Union[str, List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate HellaSwag response."""
        # Extract predicted label (0-3)
        predicted_label = self._extract_label(response, metadata)
        
        # Check if correct
        ground_truth_int = int(ground_truth) if isinstance(ground_truth, str) else ground_truth
        correct = (predicted_label == ground_truth_int)
        
        return {
            "exact_match": 1.0 if correct else 0.0,
            "predicted_label": predicted_label,
            "ground_truth": ground_truth_int,
            "correct": correct,
            "response": response,
            "evaluation_method": "hellaswag"
        }
    
    def _extract_label(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Extract predicted label (0-3) from response."""
        import re
        
        # Check if this is a placeholder response
        if "Agent" in response and "response to:" in response:
            print(f"⚠️  Detected placeholder response format!")
            return 0  # Placeholder responses default to 0
        
        response_lower = response.lower()
        
        # Priority 1: Look for "Answer: [number]" format (from our improved prompts)
        # More flexible patterns
        answer_patterns = [
            r'\*\*answer:\s*(\d+)\*\*',      # **Answer: 2**
            r'answer:\s*(\d+)',              # Answer: 2
            r'answer\s+is\s+(\d+)',         # answer is 2
            r'answer\s+=\s+(\d+)',          # answer = 2
            r'final\s+answer:\s*(\d+)',     # final answer: 2
            r'the\s+answer\s+is\s+(\d+)',   # the answer is 2
            r'correct\s+answer:\s*(\d+)',   # correct answer: 2
            r'answer\s+(\d+)',              # answer 2
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response_lower)
            if match:
                num = int(match.group(1))
                if 0 <= num <= 3:
                    print(f"✅ Extracted answer {num} using pattern: {pattern}")
                    return num
        
        # Priority 2: Look for standalone numbers (0-3) in key positions
        # Check for "Option X", "Choice X", etc.
        for i in range(4):
            if any(phrase in response_lower for phrase in [
                f"choice {i}", f"option {i}", f"label {i}",
                f"option {i})", f"choice {i})", f"#{i}"
            ]):
                return i
        
        # Priority 3: Look for A/B/C/D and map to 0-3
        if "choice a" in response_lower or "option a" in response_lower or "answer a" in response_lower:
            return 0
        elif "choice b" in response_lower or "option b" in response_lower or "answer b" in response_lower:
            return 1
        elif "choice c" in response_lower or "option c" in response_lower or "answer c" in response_lower:
            return 2
        elif "choice d" in response_lower or "option d" in response_lower or "answer d" in response_lower:
            return 3
        
        # Priority 4: Look for just a number (0-3) in the response
        # Check last few lines for standalone numbers
        lines = response.split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines (increased)
            line = line.strip()
            # Look for standalone number that's likely an answer
            # Avoid matching if it's part of a larger number or date
            num_match = re.search(r'(?:^|\s|:|,|;|\))\s*([0-3])\s*(?:$|\s|\.|,|;|\)|])', line)
            if num_match:
                num = int(num_match.group(1))
                print(f"✅ Extracted answer {num} from line: {line[:50]}")
                return num
        
        # Priority 5: Try to match response text to endings
        if metadata and "endings" in metadata:
            endings = metadata["endings"]
            for i, ending in enumerate(endings):
                # Check if response mentions this ending (first few significant words)
                ending_words = [w for w in ending.lower().split() if len(w) > 3][:3]
                if ending_words and all(word in response_lower for word in ending_words):
                    return i
        
        # Default to 0 if can't determine
        print(f"⚠️  Could not extract answer from response, defaulting to 0")
        print(f"   Response preview: {response[:200]}")
        return 0


class AlpacaEvalTask(Task):
    """
    AlpacaEval 2.0 instruction following task implementation.
    
    Evaluates instruction following capabilities.
    """
    
    def load_dataset(self):
        """Load AlpacaEval 2.0 dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            
            # Load AlpacaEval 2.0 dataset
            dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
            
            self.task_items = []
            for i, example in enumerate(dataset):
                item = TaskItem(
                    item_id=f"alpacaeval_{i}",
                    task_type=TaskType.INSTRUCTION_FOLLOWING,
                    dataset=DatasetType.ALPACAEVAL,
                    question=example["instruction"],
                    ground_truth=example.get("output", ""),  # Reference output
                    metadata={
                        "dataset": example.get("dataset", "unknown"),
                        "generator": example.get("generator", "unknown")
                    },
                    difficulty_level="medium",
                    domain="instruction_following",
                    expected_reasoning_steps=1
                )
                self.task_items.append(item)
            
            print(f"Loaded {len(self.task_items)} AlpacaEval 2.0 examples")
            
        except Exception as e:
            print(f"Error loading AlpacaEval 2.0 dataset: {e}")
            print("Falling back to placeholder data")
            self.task_items = [
                TaskItem(
                    item_id="alpacaeval_001",
                    task_type=TaskType.INSTRUCTION_FOLLOWING,
                    dataset=DatasetType.ALPACAEVAL,
                    question="Write a Python function to reverse a string.",
                    ground_truth="def reverse_string(s):\n    return s[::-1]",
                    metadata={"difficulty": "easy"},
                    difficulty_level="medium",
                    domain="instruction_following",
                    expected_reasoning_steps=1
                )
            ]
    
    def evaluate_response(
        self, 
        response: str, 
        ground_truth: Union[str, List[str]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate AlpacaEval response."""
        # For AlpacaEval, we can use reference-based metrics
        # Simple evaluation: check if response is non-empty and reasonable length
        
        # TODO: Implement more sophisticated evaluation (e.g., using LLM judge)
        response_length = len(response.strip())
        has_content = response_length > 10
        
        # Simple quality metrics
        quality_score = min(1.0, response_length / 100)  # Normalize by expected length
        
        return {
            "exact_match": 0.0,  # Not applicable for open-ended tasks
            "has_content": 1.0 if has_content else 0.0,
            "quality_score": quality_score,
            "response_length": response_length,
            "response": response,
            "ground_truth": ground_truth,
            "evaluation_method": "alpacaeval"
        }


def create_task(task_type: TaskType, dataset: DatasetType, config: Optional[TaskConfig] = None) -> Task:
    """Factory function to create tasks."""
    if config is None:
        config = TaskConfig(
            task_type=task_type,
            dataset=dataset,
            num_samples=100
        )
    
    if task_type == TaskType.MATH_REASONING:
        if dataset == DatasetType.GSM8K:
            return MathReasoningTask(config)
        elif dataset == DatasetType.MATH:
            return MathReasoningTask(config)
        else:
            raise ValueError(f"Unsupported dataset for math reasoning: {dataset}")
    elif task_type == TaskType.MULTI_HOP_QA:
        if dataset == DatasetType.HOTPOTQA:
            return MultiHopQATask(config)
        else:
            raise ValueError(f"Unsupported dataset for multi-hop QA: {dataset}")
    elif task_type == TaskType.CODE_GENERATION:
        if dataset == DatasetType.HUMANEVAL:
            return CodeGenerationTask(config)
        else:
            raise ValueError(f"Unsupported dataset for code generation: {dataset}")
    elif task_type == TaskType.COMMONSENSE_QA:
        if dataset == DatasetType.HELLASWAG:
            return HellaSwagTask(config)
        else:
            raise ValueError(f"Unsupported dataset for commonsense QA: {dataset}")
    elif task_type == TaskType.INSTRUCTION_FOLLOWING:
        if dataset == DatasetType.ALPACAEVAL:
            return AlpacaEvalTask(config)
        else:
            raise ValueError(f"Unsupported dataset for instruction following: {dataset}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")

