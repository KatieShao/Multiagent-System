"""
Quick start example for the multi-agent evaluation framework.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.experiment import run_quick_experiment
from src.core.config import create_quick_experiment
from src.baselines import CoTBaseline, SelfConsistencyBaseline, MixtureOfAgentsBaseline
from src.baselines.single_agent import BaselineConfig
from src.baselines.mixture_of_agents import MoAConfig


def main():
    """Run a quick example experiment."""
    print("🚀 Multi-Agent System Evaluation Framework - Quick Start")
    print("=" * 60)
    
    # Create a quick experiment configuration
    config = create_quick_experiment()
    print(f"📋 Experiment Configuration:")
    print(f"   - Architectures: {[arch.value for arch in config.architectures]}")
    print(f"   - Variance Levels: {[var.value for var in config.variance_levels]}")
    print(f"   - Dominance Levels: {[dom.value for dom in config.dominance_levels]}")
    print(f"   - Tasks: {config.tasks}")
    print(f"   - Datasets: {config.datasets}")
    print(f"   - Samples per condition: {config.num_samples_per_condition}")
    print()
    
    # Run the experiment
    print("🔬 Running experiment...")
    try:
        experiment_result = run_quick_experiment()
        
        print(f"✅ Experiment completed successfully!")
        print(f"   - Total results: {len(experiment_result.results)}")
        print(f"   - Duration: {experiment_result.duration:.2f} seconds")
        print(f"   - Results saved to: {experiment_result.config.output_dir}")
        print()
        
        # Show summary statistics
        print("📊 Summary Statistics:")
        for condition_id, stats in experiment_result.summary_stats.items():
            print(f"   {condition_id}:")
            print(f"     - Accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
            print(f"     - Exact Match: {stats['mean_exact_match']:.3f} ± {stats['std_exact_match']:.3f}")
            print(f"     - Trials: {stats['num_trials']}")
        print()
        
        # Show analysis results if available
        if experiment_result.analysis_results:
            print("📈 Analysis Results:")
            print("   - Statistical analysis completed")
            print("   - Results saved to analysis_results.json")
            print()
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return 1
    
    print("🎉 Quick start completed successfully!")
    return 0


async def run_baseline_example():
    """Run a simple baseline comparison example."""
    print("\n🔬 Running Baseline Comparison Example")
    print("=" * 40)
    
    # Create a simple task item for testing
    from src.core.task import TaskItem, TaskType, DatasetType
    
    task_item = TaskItem(
        item_id="example_001",
        task_type=TaskType.MATH_REASONING,
        dataset=DatasetType.GSM8K,
        question="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        ground_truth="18",
        metadata={"difficulty": "easy"}
    )
    
    # Test different baselines
    baselines = {
        "CoT": CoTBaseline(BaselineConfig()),
        "Self-Consistency": SelfConsistencyBaseline(BaselineConfig(num_samples=3)),
        "Mixture of Agents": MixtureOfAgentsBaseline(MoAConfig(num_agents=3))
    }
    
    results = {}
    for name, baseline in baselines.items():
        print(f"   Testing {name}...")
        try:
            result = await baseline.solve_task(task_item)
            results[name] = result
            print(f"     Answer: {result['final_answer']}")
            print(f"     Duration: {result['duration']:.2f}s")
        except Exception as e:
            print(f"     Error: {e}")
    
    print(f"\n📊 Baseline Comparison Results:")
    for name, result in results.items():
        if 'final_answer' in result:
            print(f"   {name}: {result['final_answer']} (took {result['duration']:.2f}s)")


if __name__ == "__main__":
    # Run main experiment
    exit_code = main()
    
    # Run baseline example
    if exit_code == 0:
        asyncio.run(run_baseline_example())
    
    sys.exit(exit_code)

