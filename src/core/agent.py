"""
Agent implementation and configuration for multi-agent systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio
import time


class AgentType(Enum):
    """Types of agents in the system."""
    DEBATER = "debater"
    JUDGE = "judge"
    ORCHESTRATOR = "orchestrator"
    EXECUTOR = "executor"
    CRITIC = "critic"
    PLANNER = "planner"
    PEER = "peer"


class VarianceLevel(Enum):
    """Agent variance levels for team composition."""
    LOW = "low"      # Same LLM, same prompt, low temperature
    MEDIUM = "medium"  # Same LLM family, different prompts/temperature
    HIGH = "high"    # Different LLMs, different prompting styles


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    agent_id: str
    agent_type: AgentType
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    prompt_template: Optional[str] = None
    role_description: Optional[str] = None
    variance_level: VarianceLevel = VarianceLevel.MEDIUM
    seed: Optional[int] = None
    
    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = 30.0
    
    # Agent-specific parameters
    confidence_threshold: float = 0.8
    max_retries: int = 3
    custom_params: Dict[str, Any] = None


class Agent(ABC):
    """
    Abstract base class for agents in multi-agent systems.
    
    Each agent can participate in different architectures and
    has configurable variance levels for team composition.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_tokens_used = 0
        self.total_time = 0.0
        
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt
            context: Additional context for the agent
            
        Returns:
            Dictionary containing response, confidence, and metadata
        """
        pass
    
    @abstractmethod
    async def evaluate_response(
        self, 
        response: str, 
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response according to given criteria.
        
        Args:
            response: The response to evaluate
            criteria: Evaluation criteria
            
        Returns:
            Dictionary containing evaluation scores and reasoning
        """
        pass
    
    def reset(self):
        """Reset agent state for new conversation."""
        self.conversation_history.clear()
        self.total_tokens_used = 0
        self.total_time = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "agent_id": self.config.agent_id,
            "total_tokens": self.total_tokens_used,
            "total_time": self.total_time,
            "avg_time_per_response": self.total_time / max(len(self.conversation_history), 1),
            "conversation_length": len(self.conversation_history)
        }


class LLMAgent(Agent):
    """
    Concrete implementation of an agent using HuggingFace transformers.
    
    Loads models from pre-downloaded HuggingFace model directory.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if self._has_cuda() else "cpu"
        self._load_model()
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Load HuggingFace model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Model path placeholder - replace with actual path on your HPC cluster
            # Example: "/path/to/qwen2.5-7b-instruct"
            model_path = self.config.custom_params.get("model_path") if self.config.custom_params else None
            
            if not model_path:
                # Default path placeholder
                model_path = "/path/to/qwen2.5-7b-instruct"  # PLACEHOLDER: Replace with actual path
            
            print(f"Loading model from: {model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device != "cuda" and self.model.device.type != "cpu":
                self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to placeholder implementation")
            self.model = None
            self.tokenizer = None
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using the loaded HuggingFace model.
        """
        start_time = time.time()
        
        if self.model is None or self.tokenizer is None:
            # Fallback to placeholder if model not loaded
            print(f"⚠️  WARNING: Model not loaded for {self.config.agent_id}, using placeholder!")
            print(f"   Model path: {self.config.custom_params.get('model_path') if self.config.custom_params else 'None'}")
            response = {
                "text": f"Agent {self.config.agent_id} response to: {prompt[:50]}...",
                "confidence": 0.8,
                "tokens_used": 100,
                "model": self.config.model_name,
                "is_placeholder": True  # Mark as placeholder
            }
        else:
            try:
                # Format prompt for Qwen instruct model
                if context and context.get("conversation_history"):
                    # Multi-turn conversation
                    messages = context["conversation_history"] + [
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                
                # Apply chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize
                import torch
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                # Calculate tokens used
                tokens_used = len(outputs[0]) - len(inputs.input_ids[0])
                
                # Simple confidence based on response length
                confidence = min(0.9, 0.5 + (len(generated_text) / 1000) * 0.4)
                
                response = {
                    "text": generated_text.strip(),
                    "confidence": confidence,
                    "tokens_used": tokens_used,
                    "model": self.config.model_name,
                    "is_placeholder": False  # Mark as real generation
                }
                
            except Exception as e:
                print(f"Error during generation: {e}")
                response = {
                    "text": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "tokens_used": 0,
                    "model": self.config.model_name
                }
        
        # Update statistics
        self.total_time += time.time() - start_time
        self.total_tokens_used += response.get("tokens_used", 0)
        
        # Store in conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "context": context
        })
        
        return response
    
    async def evaluate_response(
        self, 
        response: str, 
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using the agent's evaluation capabilities.
        
        This is a placeholder implementation.
        """
        # TODO: Implement actual evaluation logic
        evaluation = {
            "score": 0.75,
            "reasoning": f"Agent {self.config.agent_id} evaluation of response",
            "criteria_met": ["clarity", "relevance"],
            "confidence": 0.8
        }
        
        return evaluation
