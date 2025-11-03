"""
Experiment runner and management system.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from .config import ExperimentConfig, create_standard_experiment, create_quick_experiment
from .team import Team, create_team, TeamConfig
from .task import Task, create_task, TaskType, DatasetType
from .agent import Agent, LLMAgent, AgentConfig
from ..evaluation.metrics import ComprehensiveMetricsCalculator
from ..evaluation.analysis import analyze_experiment


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_id: str
    config: ExperimentConfig
    results: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    analysis_results: Optional[Dict[str, Any]] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.end_time is None:
            self.end_time = time.time()
        if self.duration is None:
            self.duration = self.end_time - self.start_time


class ExperimentRunner:
    """Main experiment runner for multi-agent evaluation."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics_calculator = ComprehensiveMetricsCalculator()
        self.results: List[Dict[str, Any]] = []
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment."""
        logger = logging.getLogger("multiagent_experiment")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def run_experiment(self) -> ExperimentResult:
        """Run the complete experiment."""
        self.logger.info(f"Starting experiment with {len(self.config.generate_conditions())} conditions")
        
        # Generate all experimental conditions
        conditions = self.config.generate_conditions()
        self.logger.info(f"Generated {len(conditions)} experimental conditions")
        
        # Run experiments for each condition
        for i, condition in enumerate(conditions):
            self.logger.info(f"Running condition {i+1}/{len(conditions)}: {condition['condition_id']}")
            
            try:
                condition_results = await self._run_condition(condition)
                self.results.extend(condition_results)
                
                if self.config.save_intermediate_results:
                    self._save_intermediate_results(condition['condition_id'], condition_results)
                    
            except Exception as e:
                self.logger.error(f"Error in condition {condition['condition_id']}: {e}")
                continue
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics()
        
        # Create experiment result
        experiment_result = ExperimentResult(
            experiment_id=f"exp_{int(time.time())}",
            config=self.config,
            results=self.results,
            summary_stats=summary_stats
        )
        
        # Save final results
        self._save_final_results(experiment_result)
        
        self.logger.info(f"Experiment completed. Total results: {len(self.results)}")
        return experiment_result
    
    async def _run_condition(self, condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run a single experimental condition."""
        condition_results = []
        
        # Create team configuration
        team_config = self.config.create_team_config(condition, f"team_{condition['condition_id']}")
        
        # Create team
        team = create_team(team_config)
        
        # Run tasks for this condition
        for task_type_str, dataset_str in zip(self.config.tasks, self.config.datasets):
            task_type = TaskType(task_type_str)
            dataset_type = DatasetType(dataset_str)
            
            # Create task
            task = create_task(task_type, dataset_type)
            
            # Get sample items
            task_items = task.get_sample_items(self.config.num_samples_per_condition)
            
            # Run team on each task item
            for item in task_items:
                try:
                    result = await self._run_single_trial(team, task, item, condition)
                    condition_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in trial for item {item.item_id}: {e}")
                    continue
        
        return condition_results
    
    async def _run_single_trial(
        self, 
        team: Team, 
        task: Task, 
        task_item: Any, 
        condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single trial of team on task item."""
        start_time = time.time()
        
        # Solve the task
        team_response = await team.solve_task(task_item.question)
        
        # Evaluate the response
        evaluation_result = task.evaluate_response(
            team_response.get("final_answer", ""),
            task_item.ground_truth
        )
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            team_response,
            task_item.ground_truth,
            {
                "task_id": task_item.item_id,
                "task_type": task_item.task_type.value,
                "dataset": task_item.dataset.value,
                "difficulty": task_item.difficulty_level
            }
        )
        
        # Get team statistics
        team_stats = team.get_team_stats()
        
        # Compile result
        result = {
            "condition_id": condition["condition_id"],
            "architecture": condition["architecture"].value,
            "variance_level": condition["variance_level"].value,
            "dominance_level": condition["dominance_level"].value,
            "task_id": task_item.item_id,
            "task_type": task_item.task_type.value,
            "dataset": task_item.dataset.value,
            "question": task_item.question,
            "ground_truth": task_item.ground_truth,
            "team_response": team_response,
            "evaluation_result": evaluation_result.to_dict() if hasattr(evaluation_result, 'to_dict') else evaluation_result,
            "metrics": metrics.to_dict(),
            "team_stats": team_stats,
            "duration": time.time() - start_time,
            "timestamp": time.time()
        }
        
        return result
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the experiment."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics by condition
        summary = {}
        
        for condition_id in df['condition_id'].unique():
            condition_data = df[df['condition_id'] == condition_id]
            
            # Extract metrics
            accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in condition_data['metrics']]
            exact_matches = [r.get('metrics', {}).get('exact_match', 0) for r in condition_data['metrics']]
            
            summary[condition_id] = {
                "num_trials": len(condition_data),
                "mean_accuracy": np.mean(accuracies) if accuracies else 0,
                "std_accuracy": np.std(accuracies) if accuracies else 0,
                "mean_exact_match": np.mean(exact_matches) if exact_matches else 0,
                "std_exact_match": np.std(exact_matches) if exact_matches else 0,
                "architecture": condition_data['architecture'].iloc[0],
                "variance_level": condition_data['variance_level'].iloc[0],
                "dominance_level": condition_data['dominance_level'].iloc[0]
            }
        
        return summary
    
    def _save_intermediate_results(self, condition_id: str, results: List[Dict[str, Any]]):
        """Save intermediate results for a condition."""
        output_dir = Path(self.config.output_dir) / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"{condition_id}_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Saved intermediate results for {condition_id}")
    
    def _save_final_results(self, experiment_result: ExperimentResult):
        """Save final experiment results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_result.results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(experiment_result.summary_stats, f, indent=2, default=str)
        
        # Save configuration
        config_file = output_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(experiment_result.config.__dict__, f, indent=2, default=str)
        
        self.logger.info(f"Saved final results to {output_dir}")
    
    def run_analysis(self, experiment_result: ExperimentResult) -> Dict[str, Any]:
        """Run statistical analysis on experiment results."""
        self.logger.info("Running statistical analysis...")
        
        # Run comprehensive analysis
        analysis_results = analyze_experiment(
            experiment_result.results,
            str(Path(self.config.output_dir) / "analysis")
        )
        
        # Update experiment result
        experiment_result.analysis_results = analysis_results
        
        # Save analysis results
        analysis_file = Path(self.config.output_dir) / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        self.logger.info("Analysis completed and saved")
        return analysis_results


def run_experiment(config: Optional[ExperimentConfig] = None) -> ExperimentResult:
    """Main function to run an experiment."""
    if config is None:
        config = create_standard_experiment()
    
    runner = ExperimentRunner(config)
    
    # Run the experiment
    experiment_result = asyncio.run(runner.run_experiment())
    
    # Run analysis
    analysis_results = runner.run_analysis(experiment_result)
    
    return experiment_result


def run_quick_experiment() -> ExperimentResult:
    """Run a quick experiment for testing."""
    config = create_quick_experiment()
    return run_experiment(config)


if __name__ == "__main__":
    # Example usage
    experiment_result = run_quick_experiment()
    print(f"Experiment completed with {len(experiment_result.results)} results")
    print(f"Results saved to {experiment_result.config.output_dir}")

