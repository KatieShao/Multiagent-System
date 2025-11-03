#!/usr/bin/env python3
"""
Test script for debate architecture on HellaSwag dataset.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.agent import AgentConfig, AgentType, VarianceLevel
from src.core.team import TeamConfig, Architecture, DominanceLevel, create_team
from src.core.task import TaskConfig, TaskType, DatasetType, create_task
from src.core.config import VarianceConfig, DominanceConfig
from src.evaluation.metrics import ComprehensiveMetricsCalculator


async def test_debate_hellaswag(model_path: str = "/path/to/qwen2.5-7b-instruct", num_samples: int = 5):
    """
    Test debate architecture on HellaSwag dataset.
    
    Args:
        model_path: Path to pre-downloaded Qwen model on HPC cluster
        num_samples: Number of HellaSwag examples to test
    """
    print("=" * 70)
    print("Testing Debate Architecture on HellaSwag Dataset")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Number of samples: {num_samples}")
    print()
    
    # Load HellaSwag dataset
    print("📚 Loading HellaSwag dataset...")
    task_config = TaskConfig(
        task_type=TaskType.COMMONSENSE_QA,
        dataset=DatasetType.HELLASWAG,
        num_samples=num_samples
    )
    task = create_task(TaskType.COMMONSENSE_QA, DatasetType.HELLASWAG, task_config)
    task_items = task.get_sample_items(num_samples)
    print(f"✅ Loaded {len(task_items)} HellaSwag examples\n")
    
    # Create debate team with 3 debaters + 1 judge
    print("👥 Creating debate team...")
    
    # Create agent configs - all using the same model with different temperatures
    agent_configs = []
    
    # Debaters with different roles and temperatures for variance
    debater_roles = [
        {
            "id": "debater_0",
            "role": "Logical reasoning expert - analyzes causal relationships and logical connections",
            "temperature": 0.3  # Lower for more focused reasoning
        },
        {
            "id": "debater_1", 
            "role": "Commonsense knowledge specialist - uses real-world understanding and everyday scenarios",
            "temperature": 0.5  # Medium for balanced reasoning
        },
        {
            "id": "debater_2",
            "role": "Language understanding expert - focuses on semantic coherence and linguistic fit",
            "temperature": 0.7  # Higher for more creative connections
        }
    ]
    
    for debater_info in debater_roles:
        config = AgentConfig(
            agent_id=debater_info["id"],
            agent_type=AgentType.DEBATER,
            model_name="qwen2.5-7b-instruct",
            temperature=debater_info["temperature"],
            max_tokens=300,  # Increased for better reasoning
            prompt_template="standard",
            role_description=debater_info["role"],
            variance_level=VarianceLevel.LOW,
            seed=42 + int(debater_info["id"].split("_")[1]),
            custom_params={"model_path": model_path}
        )
        agent_configs.append(config)
    
    # Judge
    judge_config = AgentConfig(
        agent_id="judge",
        agent_type=AgentType.JUDGE,
        model_name="qwen2.5-7b-instruct",
        temperature=0.1,  # Low temperature for consistent judging
        max_tokens=150,  # Increased for better evaluation
        prompt_template="judge",
        role_description="Expert judge who evaluates all arguments and makes the final decision based on reasoning quality",
        variance_level=VarianceLevel.LOW,
        seed=100,
        custom_params={"model_path": model_path}
    )
    agent_configs.append(judge_config)
    
    # Create team config
    team_config = TeamConfig(
        team_id="debate_team_hellaswag",
        architecture=Architecture.DEBATE_VOTE,
        dominance_level=DominanceLevel.NONE,
        variance_level=VarianceLevel.LOW,
        agent_configs=agent_configs,
        max_rounds=1,
        consensus_threshold=0.8,
        debate_rounds=1,
        voting_strategy="majority",
        early_stopping=True
    )
    
    # Create team
    team = create_team(team_config)
    print(f"✅ Created debate team with {len(agent_configs)} agents\n")
    
    # Metrics calculator
    metrics_calc = ComprehensiveMetricsCalculator()
    
    # Run on test examples
    print("🔬 Running debate on HellaSwag examples...")
    print("-" * 70)
    
    results = []
    correct = 0
    
    for idx, item in enumerate(task_items, 1):
        print(f"\n📝 Example {idx}/{len(task_items)}")
        print(f"Context: {item.metadata.get('context', item.question[:100])}...")
        
        try:
            # Prepare context for the team
            task_context = {
                "task_type": "commonsense_qa",
                "dataset": "hellaswag",
                "item_id": item.item_id
            }
            
            # Run debate
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
                    "task_type": item.task_type.value,
                    "dataset": item.dataset.value
                }
            )
            
            is_correct = evaluation.get("correct", False)
            predicted = evaluation.get("predicted_label", -1)
            ground_truth = evaluation.get("ground_truth", -1)
            
            if is_correct:
                correct += 1
            
            results.append({
                "item_id": item.item_id,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "accuracy": metrics.accuracy,
                "response": final_answer[:100] + "..." if len(final_answer) > 100 else final_answer
            })
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"Predicted: {predicted}, Ground Truth: {ground_truth} - {status}")
            print(f"Response preview: {final_answer[:150]}...")
            print(f"Full judge response length: {len(final_answer)} chars")
            
            # Show debater responses too
            individual_responses = team_response.get("individual_responses", {})
            print(f"\nDebater responses:")
            for agent_id, resp in individual_responses.items():
                resp_text = resp.get('text', '')[:100] if isinstance(resp, dict) else str(resp)[:100]
                print(f"  {agent_id}: {resp_text}...")
            
        except Exception as e:
            print(f"❌ Error on example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Results Summary")
    print("=" * 70)
    print(f"Total examples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / len(results) * 100:.2f}%")
    print()
    
    # Individual results
    print("Detailed Results:")
    for result in results:
        status = "✅" if result["correct"] else "❌"
        print(f"{status} {result['item_id']}: Predicted {result['predicted']}, "
              f"Ground Truth {result['ground_truth']}")
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)
    
    return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test debate architecture on HellaSwag")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/path/to/qwen2.5-7b-instruct",
        help="Path to pre-downloaded Qwen model (PLACEHOLDER - update with actual path)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of HellaSwag examples to test"
    )
    
    args = parser.parse_args()
    
    if args.model_path == "/path/to/qwen2.5-7b-instruct":
        print("⚠️  WARNING: Using placeholder model path!")
        print("Please update --model-path with your actual model path on the HPC cluster")
        print("Example: --model-path /gpfs/path/to/qwen2.5-7b-instruct")
        print()
    
    # Run test
    results = asyncio.run(test_debate_hellaswag(
        model_path=args.model_path,
        num_samples=args.num_samples
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
