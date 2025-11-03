#!/usr/bin/env python3
"""
Unified evaluation runner for multi-agent systems.

Supports all benchmarks, architectures, and single/multi-model modes.
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent import AgentConfig, AgentType, VarianceLevel
from src.core.team import Architecture, DominanceLevel, TeamConfig, create_team
from src.core.team_utils import create_multi_model_team, create_qwen_multi_model_team
from src.core.task import TaskConfig, TaskType, DatasetType, create_task
from src.evaluation.metrics import ComprehensiveMetricsCalculator


# Benchmark to task type and dataset mapping
BENCHMARK_CONFIG = {
    "gsm8k": {
        "task_type": TaskType.MATH_REASONING,
        "dataset": DatasetType.GSM8K
    },
    "math": {
        "task_type": TaskType.MATH_REASONING,
        "dataset": DatasetType.MATH
    },
    "hotpotqa": {
        "task_type": TaskType.MULTI_HOP_QA,
        "dataset": DatasetType.HOTPOTQA
    },
    "humaneval": {
        "task_type": TaskType.CODE_GENERATION,
        "dataset": DatasetType.HUMANEVAL
    },
    "hellaswag": {
        "task_type": TaskType.COMMONSENSE_QA,
        "dataset": DatasetType.HELLASWAG
    },
    "alpacaeval": {
        "task_type": TaskType.INSTRUCTION_FOLLOWING,
        "dataset": DatasetType.ALPACAEVAL
    }
}

# Architecture mapping
ARCHITECTURE_MAP = {
    "debate": Architecture.DEBATE_VOTE,
    "debate_vote": Architecture.DEBATE_VOTE,
    "orchestrator": Architecture.ORCHESTRATOR_SUBAGENTS,
    "orchestrator_subagents": Architecture.ORCHESTRATOR_SUBAGENTS,
    "roleplay": Architecture.ROLE_PLAY_TEAMWORK,
    "role_play_teamwork": Architecture.ROLE_PLAY_TEAMWORK,
    "role_play": Architecture.ROLE_PLAY_TEAMWORK
}


async def run_evaluation(
    benchmark: str,
    architecture: str,
    model_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    use_multi_model: bool = False,
    num_samples: int = 10,
    dominance_level: str = "none",
    output_dir: str = "results"
):
    """
    Run evaluation on a benchmark with specified architecture.
    
    Args:
        benchmark: Benchmark name (gsm8k, math, hotpotqa, humaneval, hellaswag, alpacaeval)
        architecture: Architecture name (debate, orchestrator, roleplay)
        model_path: Path to single model (single-model mode)
        base_model_path: Base path for models (multi-model mode)
        use_multi_model: Whether to use multiple models
        num_samples: Number of samples to evaluate
        dominance_level: Dominance level (none, moderate, strong)
        output_dir: Output directory for results
    """
    print("=" * 80)
    print(f"Multi-Agent System Evaluation")
    print("=" * 80)
    print(f"Benchmark: {benchmark}")
    print(f"Architecture: {architecture}")
    print(f"Mode: {'Multi-model' if use_multi_model else 'Single-model'}")
    print(f"Num samples: {num_samples}")
    print(f"Dominance level: {dominance_level}")
    print("=" * 80)
    print()
    
    # Validate benchmark
    if benchmark.lower() not in BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_CONFIG.keys())}")
    
    # Validate architecture
    arch_lower = architecture.lower()
    if arch_lower not in ARCHITECTURE_MAP:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(ARCHITECTURE_MAP.keys())}")
    
    # Get task configuration
    task_config_map = BENCHMARK_CONFIG[benchmark.lower()]
    task_type = task_config_map["task_type"]
    dataset_type = task_config_map["dataset"]
    
    # Load dataset
    print(f"📚 Loading {benchmark.upper()} dataset...")
    task_config = TaskConfig(
        task_type=task_type,
        dataset=dataset_type,
        num_samples=num_samples
    )
    task = create_task(task_type, dataset_type, task_config)
    task_items = task.get_sample_items(num_samples)
    print(f"✅ Loaded {len(task_items)} examples\n")
    
    # Create team
    print(f"👥 Creating {architecture} team...")
    
    # Map dominance level
    dominance_map = {
        "none": DominanceLevel.NONE,
        "moderate": DominanceLevel.MODERATE,
        "strong": DominanceLevel.STRONG
    }
    dom_level = dominance_map.get(dominance_level.lower(), DominanceLevel.NONE)
    arch_enum = ARCHITECTURE_MAP[arch_lower]
    
    if use_multi_model:
        # Multi-model mode
        if not base_model_path:
            raise ValueError("Multi-model mode requires --base-model-path")
        
        print(f"📦 Using multiple models from: {base_model_path}")
        
        # Create team with multiple models
        if architecture.lower() in ["debate", "debate_vote"]:
            # Debate architecture: 3 debaters + 1 judge
            team_config = create_qwen_multi_model_team(
                team_id=f"{architecture}_{benchmark}_team",
                architecture=arch_enum,
                dominance_level=dom_level,
                model_sizes=["3b", "7b", "32b", "72b"],
                base_model_path=base_model_path,
                team_composition={
                    "debater_count": 3,
                    "judge_model": "72b",
                    "temperatures": [0.3, 0.5, 0.7]
                }
            )
        elif architecture.lower() in ["orchestrator", "orchestrator_subagents"]:
            # Orchestrator: 1 orchestrator + 3 sub-agents (planner, critic, executor)
            # Create custom model configs for orchestrator architecture
            model_configs = []
            
            # Orchestrator (largest model)
            model_configs.append({
                "agent_id": "orchestrator",
                "agent_type": AgentType.ORCHESTRATOR,
                "model_name": "qwen2.5-72b-instruct",
                "model_path": f"{base_model_path}/qwen2.5-72b-instruct",
                "temperature": 0.1,
                "max_tokens": 200,
                "role_description": "Orchestrator coordinating the team"
            })
            
            # Sub-agents (different sizes)
            sub_agent_configs = [
                {"id": "planner", "type": AgentType.PLANNER, "size": "3b", "temp": 0.3},
                {"id": "critic", "type": AgentType.CRITIC, "size": "7b", "temp": 0.5},
                {"id": "executor", "type": AgentType.EXECUTOR, "size": "32b", "temp": 0.7}
            ]
            
            for sub_config in sub_agent_configs:
                model_configs.append({
                    "agent_id": sub_config["id"],
                    "agent_type": sub_config["type"],
                    "model_name": f"qwen2.5-{sub_config['size']}-instruct",
                    "model_path": f"{base_model_path}/qwen2.5-{sub_config['size']}-instruct",
                    "temperature": sub_config["temp"],
                    "max_tokens": 300,
                    "role_description": f"{sub_config['id'].capitalize()} agent"
                })
            
            team_config = create_multi_model_team(
                team_id=f"{architecture}_{benchmark}_team",
                architecture=arch_enum,
                dominance_level=dom_level,
                model_configs=model_configs
            )
        else:  # roleplay
            # Role-play: 3 peers
            model_configs = []
            
            peer_configs = [
                {"id": "peer_0", "size": "3b", "temp": 0.3},
                {"id": "peer_1", "size": "7b", "temp": 0.5},
                {"id": "peer_2", "size": "32b", "temp": 0.7}
            ]
            
            for peer_config in peer_configs:
                model_configs.append({
                    "agent_id": peer_config["id"],
                    "agent_type": AgentType.PEER,
                    "model_name": f"qwen2.5-{peer_config['size']}-instruct",
                    "model_path": f"{base_model_path}/qwen2.5-{peer_config['size']}-instruct",
                    "temperature": peer_config["temp"],
                    "max_tokens": 300,
                    "role_description": f"Peer collaborator"
                })
            
            team_config = create_multi_model_team(
                team_id=f"{architecture}_{benchmark}_team",
                architecture=arch_enum,
                dominance_level=dom_level,
                model_configs=model_configs
            )
        
        team = create_team(team_config)
        print(f"✅ Created multi-model team with {len(team_config.agent_configs)} agents")
        print(f"   Models: {[cfg.model_name for cfg in team_config.agent_configs]}\n")
    else:
        # Single-model mode
        if not model_path:
            raise ValueError("Single-model mode requires --model-path")
        
        print(f"📦 Using single model: {model_path}")
        
        # Create team config manually
        agent_configs = []
        
        if architecture.lower() in ["debate", "debate_vote"]:
            # 3 debaters + 1 judge
            for i in range(3):
                agent_configs.append(AgentConfig(
                    agent_id=f"debater_{i}",
                    agent_type=AgentType.DEBATER,
                    model_name="qwen2.5-7b-instruct",
                    model_path=model_path,
                    temperature=0.3 + i * 0.2,  # 0.3, 0.5, 0.7
                    max_tokens=300,
                    role_description=f"Debater {i+1}",
                    variance_level=VarianceLevel.LOW
                ))
            
            agent_configs.append(AgentConfig(
                agent_id="judge",
                agent_type=AgentType.JUDGE,
                model_name="qwen2.5-7b-instruct",
                model_path=model_path,
                temperature=0.1,
                max_tokens=150,
                role_description="Judge",
                variance_level=VarianceLevel.LOW
            ))
        
        elif architecture.lower() in ["orchestrator", "orchestrator_subagents"]:
            # 1 orchestrator + 3 sub-agents (planner, critic, executor)
            agent_configs.append(AgentConfig(
                agent_id="orchestrator",
                agent_type=AgentType.ORCHESTRATOR,
                model_name="qwen2.5-7b-instruct",
                model_path=model_path,
                temperature=0.1,
                max_tokens=200,
                role_description="Orchestrator coordinating the team",
                variance_level=VarianceLevel.LOW
            ))
            
            agent_configs.append(AgentConfig(
                agent_id="planner",
                agent_type=AgentType.PLANNER,
                model_name="qwen2.5-7b-instruct",
                model_path=model_path,
                temperature=0.3,
                max_tokens=300,
                role_description="Planner creating detailed plans",
                variance_level=VarianceLevel.LOW
            ))
            
            agent_configs.append(AgentConfig(
                agent_id="critic",
                agent_type=AgentType.CRITIC,
                model_name="qwen2.5-7b-instruct",
                model_path=model_path,
                temperature=0.5,
                max_tokens=200,
                role_description="Critic reviewing plans and executions",
                variance_level=VarianceLevel.LOW
            ))
            
            agent_configs.append(AgentConfig(
                agent_id="executor",
                agent_type=AgentType.EXECUTOR,
                model_name="qwen2.5-7b-instruct",
                model_path=model_path,
                temperature=0.7,
                max_tokens=300,
                role_description="Executor implementing plans",
                variance_level=VarianceLevel.LOW
            ))
        
        else:  # roleplay
            # 3 peers
            for i in range(3):
                agent_configs.append(AgentConfig(
                    agent_id=f"peer_{i}",
                    agent_type=AgentType.PEER,
                    model_name="qwen2.5-7b-instruct",
                    model_path=model_path,
                    temperature=0.3 + i * 0.2,  # 0.3, 0.5, 0.7
                    max_tokens=300,
                    role_description=f"Peer collaborator {i+1}",
                    variance_level=VarianceLevel.LOW
                ))
        
        team_config = TeamConfig(
            team_id=f"{architecture}_{benchmark}_team",
            architecture=arch_enum,
            dominance_level=dom_level,
            variance_level=VarianceLevel.LOW if not use_multi_model else VarianceLevel.HIGH,
            agent_configs=agent_configs,
            max_rounds=10,
            consensus_threshold=0.8,
            debate_rounds=3,
            voting_strategy="majority",
            early_stopping=True
        )
        
        team = create_team(team_config)
        print(f"✅ Created single-model team with {len(agent_configs)} agents\n")
    
    # Metrics calculator
    metrics_calc = ComprehensiveMetricsCalculator()
    
    # Run evaluation
    print(f"🔬 Running evaluation on {benchmark.upper()}...")
    print("-" * 80)
    
    results = []
    correct = 0
    total = 0
    
    for idx, item in enumerate(task_items, 1):
        print(f"\n📝 Example {idx}/{len(task_items)}")
        if benchmark in ["gsm8k", "math"]:
            print(f"Question: {item.question[:100]}...")
        elif benchmark == "hellaswag":
            print(f"Context: {item.metadata.get('context', item.question[:100])}...")
        else:
            print(f"Task: {item.question[:100]}...")
        
        try:
            # Prepare context
            task_context = {
                "task_type": task_type.value,
                "dataset": dataset_type.value,
                "item_id": item.item_id
            }
            
            # Run team on task
            team_response = await team.solve_task(item.question, task_context)
            final_answer = team_response.get("final_answer", "")
            
            # Evaluate
            evaluation = task.evaluate_response(
                final_answer,
                item.ground_truth,
                item.metadata
            )
            
            # Calculate metrics
            metrics = metrics_calc.calculate_all_metrics(
                team_response,
                item.ground_truth,
                {
                    "task_id": item.item_id,
                    "task_type": task_type.value,
                    "dataset": dataset_type.value
                }
            )
            
            # Determine correctness based on evaluation method
            if benchmark in ["gsm8k", "math"]:
                is_correct = evaluation.get("numerical_match", False) or evaluation.get("exact_match", False)
            elif benchmark == "hellaswag":
                is_correct = evaluation.get("correct", False)
            elif benchmark == "hotpotqa":
                is_correct = evaluation.get("exact_match", False) or evaluation.get("f1_score", 0) > 0.8
            elif benchmark == "humaneval":
                is_correct = evaluation.get("test_results", {}).get("pass_at_1", False)
            else:  # alpacaeval
                is_correct = evaluation.get("has_content", False) and evaluation.get("quality_score", 0) > 0.5
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "item_id": item.item_id,
                "correct": is_correct,
                "evaluation": evaluation,
                "metrics": metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics,
                "response": final_answer[:200]
            })
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"Result: {status}")
            print(f"Response: {final_answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Error on example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Evaluation Summary")
    print("=" * 80)
    print(f"Benchmark: {benchmark.upper()}")
    print(f"Architecture: {architecture}")
    print(f"Mode: {'Multi-model' if use_multi_model else 'Single-model'}")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    if total > 0:
        print(f"Accuracy: {correct / total * 100:.2f}%")
    print("=" * 80)
    
    # Save results
    import json
    from pathlib import Path as PathLib
    
    output_path = PathLib(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{benchmark}_{architecture}_{'multi' if use_multi_model else 'single'}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "benchmark": benchmark,
            "architecture": architecture,
            "mode": "multi-model" if use_multi_model else "single-model",
            "num_samples": num_samples,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "results": results
        }, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {results_file}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation runner for multi-agent systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, debate architecture, GSM8K
  python run_evaluation.py --benchmark gsm8k --architecture debate \\
      --model-path /path/to/qwen2.5-7b-instruct
  
  # Multi-model, orchestrator architecture, HellaSwag
  python run_evaluation.py --benchmark hellaswag --architecture orchestrator \\
      --base-model-path /gpfs/models/qwen2.5 --multi-model
  
  # Role-play, MATH dataset, multi-model
  python run_evaluation.py --benchmark math --architecture roleplay \\
      --base-model-path /gpfs/models/qwen2.5 --multi-model --num-samples 20
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark to evaluate (gsm8k, math, hotpotqa, humaneval, hellaswag, alpacaeval)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Team architecture (debate, orchestrator, roleplay)"
    )
    
    # Model configuration
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Path to single model (single-model mode)"
    )
    model_group.add_argument(
        "--base-model-path",
        type=str,
        help="Base path for models (multi-model mode, e.g., /gpfs/models/qwen2.5)"
    )
    
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Use multiple different models (requires --base-model-path)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--dominance-level",
        type=str,
        default="none",
        choices=["none", "moderate", "strong"],
        help="Leader dominance level (default: none)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    use_multi_model = args.multi_model or args.base_model_path is not None
    
    if use_multi_model and not args.base_model_path:
        print("❌ ERROR: --multi-model requires --base-model-path")
        return 1
    
    if not use_multi_model and not args.model_path:
        print("❌ ERROR: Single-model mode requires --model-path")
        return 1
    
    # Run evaluation
    try:
        results = asyncio.run(run_evaluation(
            benchmark=args.benchmark,
            architecture=args.architecture,
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            use_multi_model=use_multi_model,
            num_samples=args.num_samples,
            dominance_level=args.dominance_level,
            output_dir=args.output_dir
        ))
        return 0
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
