"""
Benchmark-specific prompts and role specifications for multi-agent systems.

This module contains all prompt templates and role definitions for different
benchmarks (GSM8K, MATH, HotpotQA, HumanEval, HellaSwag, AlpacaEval).
"""

from typing import Dict, List, Any


def get_benchmark_specific_prompts(dataset: str, task: str) -> Dict[str, Any]:
    """
    Generate benchmark-specific prompts for debaters and judge.
    
    Args:
        dataset: Dataset name (gsm8k, math, hotpotqa, humaneval, hellaswag, alpacaeval)
        task: The task/question text
        
    Returns:
        Dictionary containing:
            - roles_and_approaches: List of role/approach dicts for debaters
            - debater_base_prompt: Template for debater prompts
            - judge_base_prompt: Template for judge prompts
    """
    dataset_lower = dataset.lower() if dataset else ""
    
    if dataset_lower in ["gsm8k", "math"]:
        return _get_math_reasoning_prompts()
    elif dataset_lower == "hotpotqa":
        return _get_hotpotqa_prompts()
    elif dataset_lower == "humaneval":
        return _get_humaneval_prompts()
    elif dataset_lower == "hellaswag":
        return _get_hellaswag_prompts()
    elif dataset_lower == "alpacaeval":
        return _get_alpacaeval_prompts()
    else:
        return _get_default_prompts()


def _get_math_reasoning_prompts() -> Dict[str, Any]:
    """Get prompts for math reasoning tasks (GSM8K, MATH)."""
    roles_and_approaches = [
        {
            "role": "You are a mathematical reasoning expert",
            "approach": "Break down the problem into smaller steps. Identify the key mathematical operations needed. Show your work step by step with clear calculations."
        },
        {
            "role": "You are a problem-solving strategist",
            "approach": "Analyze the problem structure. Identify what information is given and what needs to be found. Consider different solution approaches and choose the most efficient one."
        },
        {
            "role": "You are a computational accuracy specialist",
            "approach": "Focus on precise calculations and numerical accuracy. Double-check arithmetic operations and ensure all steps are mathematically correct. Verify your final answer makes sense."
        }
    ]
    
    debater_base_prompt = """{role}. You are solving a mathematical reasoning problem.

{task}

{approach}

Solve this problem step by step:
1. Read the problem carefully and identify what is being asked
2. Break down the problem into smaller, manageable steps
3. Show all your calculations clearly
4. Verify your answer makes sense

Provide your final answer as a number. If multiple steps are needed, show your work."""
    
    judge_base_prompt = """You are an expert judge evaluating mathematical reasoning solutions.

**Original Problem:**
{task}

**Agent Solutions:**
{all_responses}

Your task:
1. Review each agent's solution approach and mathematical reasoning
2. Check the correctness of calculations and logical steps
3. Identify which agent provided the most accurate and well-reasoned solution
4. Synthesize insights from multiple agents if needed
5. Provide the final correct answer

**IMPORTANT:** For math problems, your final answer should be a single number. Show your reasoning if needed, but end with just the numerical answer."""
    
    return {
        "roles_and_approaches": roles_and_approaches,
        "debater_base_prompt": debater_base_prompt,
        "judge_base_prompt": judge_base_prompt
    }


def _get_hotpotqa_prompts() -> Dict[str, Any]:
    """Get prompts for multi-hop question answering (HotpotQA)."""
    roles_and_approaches = [
        {
            "role": "You are a fact-finding expert",
            "approach": "Break down the question into sub-questions. Identify what information needs to be found at each step. Trace the logical connections between different pieces of information."
        },
        {
            "role": "You are an information synthesis specialist",
            "approach": "Analyze how different facts relate to each other. Consider multiple sources of information and how they combine to answer the question. Look for logical chains of reasoning."
        },
        {
            "role": "You are a verification and accuracy expert",
            "approach": "Verify each piece of information used. Ensure the reasoning chain is logically sound. Cross-check that the final answer follows from the intermediate facts."
        }
    ]
    
    debater_base_prompt = """{role}. You are solving a multi-hop question answering task.

{task}

{approach}

To answer this question:
1. Identify what information is needed to answer the question
2. Break down the question into sub-questions if needed
3. Trace the logical connections between different pieces of information
4. Synthesize the information to arrive at the final answer

Provide a clear, concise answer based on your reasoning."""
    
    judge_base_prompt = """You are an expert judge evaluating multi-hop question answering responses.

**Original Question:**
{task}

**Agent Responses:**
{all_responses}

Your task:
1. Review each agent's reasoning chain and information synthesis
2. Evaluate the logical soundness of their multi-hop reasoning
3. Check if all necessary information is properly connected
4. Identify which agent provided the most accurate and well-reasoned answer
5. Synthesize the best reasoning from all agents if needed
6. Provide the final answer

Provide a clear, concise final answer."""
    
    return {
        "roles_and_approaches": roles_and_approaches,
        "debater_base_prompt": debater_base_prompt,
        "judge_base_prompt": judge_base_prompt
    }


def _get_humaneval_prompts() -> Dict[str, Any]:
    """Get prompts for code generation (HumanEval)."""
    roles_and_approaches = [
        {
            "role": "You are a code implementation expert",
            "approach": "Analyze the function signature and requirements carefully. Implement the solution step by step. Ensure your code handles edge cases and follows best practices."
        },
        {
            "role": "You are an algorithm design specialist",
            "approach": "Consider different algorithmic approaches. Choose the most efficient solution. Ensure the logic is correct and handles all test cases."
        },
        {
            "role": "You are a code quality and correctness expert",
            "approach": "Focus on code correctness, readability, and edge case handling. Verify the implementation matches the requirements exactly. Ensure the code will pass all test cases."
        }
    ]
    
    debater_base_prompt = """{role}. You are solving a code generation task.

{task}

{approach}

To implement this function:
1. Read the function signature and docstring carefully
2. Understand the requirements and expected behavior
3. Consider edge cases and test scenarios
4. Implement the solution with clear, correct code
5. Ensure your code matches the requirements exactly

Provide your complete implementation."""
    
    judge_base_prompt = """You are an expert judge evaluating code generation solutions.

**Original Task:**
{task}

**Agent Implementations:**
{all_responses}

Your task:
1. Review each agent's code implementation and approach
2. Evaluate correctness, efficiency, and code quality
3. Check if implementations handle edge cases properly
4. Identify which agent provided the best solution
5. Synthesize the best code from all agents if needed
6. Provide the final, correct implementation

Provide your final code implementation."""
    
    return {
        "roles_and_approaches": roles_and_approaches,
        "debater_base_prompt": debater_base_prompt,
        "judge_base_prompt": judge_base_prompt
    }


def _get_hellaswag_prompts() -> Dict[str, Any]:
    """Get prompts for commonsense reasoning (HellaSwag)."""
    roles_and_approaches = [
        {
            "role": "You are a logical reasoning expert",
            "approach": "Analyze the logical connections and causal relationships between the context and each option. Think step-by-step about what would naturally follow."
        },
        {
            "role": "You are a commonsense knowledge specialist",
            "approach": "Use your understanding of how the world works. Consider everyday scenarios, human behavior, and physical laws. Think about what makes the most sense in context."
        },
        {
            "role": "You are a language understanding expert",
            "approach": "Focus on linguistic coherence and semantic fit. Analyze how well each option connects to the context linguistically and meaningfully."
        }
    ]
    
    debater_base_prompt = """{role}. You are solving a commonsense reasoning task.

{task}

{approach}

For each option, briefly explain:
- Why it does or doesn't make sense given the context
- How it fits (or doesn't fit) with commonsense understanding

After analyzing all options, provide your answer in this exact format:
**Answer: [number]**

Where [number] is 0, 1, 2, or 3 corresponding to the best option."""
    
    judge_base_prompt = """You are an expert judge evaluating responses to a multiple-choice commonsense reasoning task.

**Original Task:**
{task}

**Agent Responses:**
{all_responses}

Your task:
1. Review each agent's reasoning and answer choice
2. Consider which agent provided the most sound logical analysis
3. Synthesize the best reasoning from all agents if needed
4. Make the final decision

**IMPORTANT:** You must output your answer in this exact format:
**Answer: [number]**

Where [number] is 0, 1, 2, or 3 corresponding to the best option. Only output the number, nothing else."""
    
    return {
        "roles_and_approaches": roles_and_approaches,
        "debater_base_prompt": debater_base_prompt,
        "judge_base_prompt": judge_base_prompt
    }


def _get_alpacaeval_prompts() -> Dict[str, Any]:
    """Get prompts for instruction following (AlpacaEval)."""
    roles_and_approaches = [
        {
            "role": "You are an instruction understanding expert",
            "approach": "Carefully read and understand the instruction. Identify all requirements and constraints. Follow the instruction precisely and completely."
        },
        {
            "role": "You are a task completion specialist",
            "approach": "Break down the instruction into actionable steps. Ensure all aspects of the instruction are addressed. Provide a complete and accurate response."
        },
        {
            "role": "You are a quality and accuracy expert",
            "approach": "Focus on producing high-quality, accurate output that fully satisfies the instruction. Verify completeness and correctness of the response."
        }
    ]
    
    debater_base_prompt = """{role}. You are following an instruction.

{task}

{approach}

To complete this task:
1. Read the instruction carefully and understand all requirements
2. Identify what needs to be done
3. Break down the task into steps if needed
4. Provide a complete, accurate response that fulfills the instruction

Provide your response."""
    
    judge_base_prompt = """You are an expert judge evaluating instruction following responses.

**Original Instruction:**
{task}

**Agent Responses:**
{all_responses}

Your task:
1. Review how well each agent followed the instruction
2. Evaluate completeness, accuracy, and quality of each response
3. Identify which agent provided the best response
4. Synthesize the best elements from all agents if needed
5. Provide the final, best response

Provide your final response that best follows the instruction."""
    
    return {
        "roles_and_approaches": roles_and_approaches,
        "debater_base_prompt": debater_base_prompt,
        "judge_base_prompt": judge_base_prompt
    }


def _get_default_prompts() -> Dict[str, Any]:
    """Get default generic prompts for unknown benchmarks."""
    roles_and_approaches = [
        {
            "role": "You are a problem-solving expert",
            "approach": "Analyze the problem carefully and think through the solution step by step."
        },
        {
            "role": "You are a reasoning specialist",
            "approach": "Use logical reasoning and systematic thinking to solve the problem."
        },
        {
            "role": "You are an accuracy expert",
            "approach": "Focus on correctness and precision in your solution."
        }
    ]
    
    debater_base_prompt = """{role}. You are solving the following task:

{task}

{approach}

Provide your best answer with clear reasoning."""
    
    judge_base_prompt = """You are an expert judge evaluating multiple responses to the following task:

{task}

**Agent Responses:**
{all_responses}

Your task:
1. Review each agent's reasoning and answer choice
2. Consider which agent provided the most sound logical analysis
3. Synthesize the best reasoning from all agents if needed
4. Make the final decision

Provide the best final answer."""
    
    return {
        "roles_and_approaches": roles_and_approaches,
        "debater_base_prompt": debater_base_prompt,
        "judge_base_prompt": judge_base_prompt
    }

