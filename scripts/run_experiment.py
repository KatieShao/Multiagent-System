#!/usr/bin/env python3
"""
Script to run multi-agent evaluation experiments.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.experiment import run_experiment, run_quick_experiment
from src.core.config import create_standard_experiment, create_quick_experiment


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run multi-agent evaluation experiments")
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "standard", "custom"], 
        default="quick",
        help="Experiment mode to run"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=None,
        help="Number of samples per condition (overrides default)"
    )
    
    parser.add_argument(
        "--num-replications", 
        type=int, 
        default=None,
        help="Number of replications (overrides default)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Multi-Agent System Evaluation Framework")
    print(f"Mode: {args.mode}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        if args.mode == "quick":
            config = create_quick_experiment()
            if args.num_samples:
                config.num_samples_per_condition = args.num_samples
            if args.num_replications:
                config.num_replications = args.num_replications
            config.output_dir = args.output_dir
            config.log_level = args.log_level
            
            experiment_result = run_experiment(config)
            
        elif args.mode == "standard":
            config = create_standard_experiment()
            if args.num_samples:
                config.num_samples_per_condition = args.num_samples
            if args.num_replications:
                config.num_replications = args.num_replications
            config.output_dir = args.output_dir
            config.log_level = args.log_level
            
            experiment_result = run_experiment(config)
            
        else:  # custom
            print("Custom mode not yet implemented. Use quick or standard mode.")
            return 1
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"   - Total results: {len(experiment_result.results)}")
        print(f"   - Duration: {experiment_result.duration:.2f} seconds")
        print(f"   - Results saved to: {experiment_result.config.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

