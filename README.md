# Multi-Agent System Evaluation Framework

A comprehensive framework for evaluating different multi-agent system architectures and their performance across various task types and team compositions.

## 🎯 Research Goal

**Test whether higher variance teams benefit more from a dominant leader than low-variance teams.**

This framework implements a 3×3×3 factorial design to systematically evaluate:

- **3 Architectures**: Debate+Vote, Orchestrator→Sub-agents, Role-play Teamwork
- **3 Variance Levels**: Low, Medium, High (agent diversity)
- **3 Dominance Levels**: None, Moderate, Strong (leadership influence)

## 🏗️ Framework Architecture

### Core Components

```
src/
├── core/                    # Core framework components
│   ├── agent.py            # Agent implementations and configurations
│   ├── team.py             # Team architectures and dominance patterns
│   ├── task.py             # Task and dataset interfaces
│   ├── config.py           # Experiment configuration system
│   └── experiment.py       # Experiment runner and management
├── evaluation/             # Evaluation metrics and analysis
│   ├── metrics.py          # Performance and process metrics
│   ├── analysis.py         # Statistical analysis tools
│   └── diversity.py        # Diversity and variance calculations
└── baselines/              # Baseline implementations
    ├── single_agent.py     # Single-agent baselines (CoT, Self-Consistency)
    └── mixture_of_agents.py # Mixture of Agents baseline
```

### Key Features

- **Modular Design**: Easy to extend with new architectures, tasks, and metrics
- **Comprehensive Evaluation**: Task performance, process metrics, and diversity measures
- **Statistical Analysis**: Mixed-effects modeling and interaction analysis
- **Baseline Comparisons**: Single-agent and MoA baselines for reference
- **Reproducible**: Fixed seeds and detailed logging for reproducibility

## 🚀 Quick Start

### Installation

**Python Version Requirement**: Python 3.9 or higher is required.

#### Option 1: Using Conda (Recommended for Scientific Computing)

```bash
# Clone the repository
git clone <repository-url>
cd Multiagent-System

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate multiagent-system

# Verify installation
python --version  # Should show Python 3.9+
```

See [CONDA_SETUP.md](CONDA_SETUP.md) for detailed conda instructions.

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd Multiagent-System

# Verify Python version (should be 3.9+)
python --version  # or python3 --version

# Install dependencies
pip install -r requirements.txt

# Optional: Install project in editable mode
pip install -e .
```

### Run a Quick Experiment

```bash
# Run a quick experiment (reduced conditions for testing)
python examples/quick_start.py

# Or use the experiment script
python scripts/run_experiment.py --mode quick --output-dir results/quick
```

### Run Full Experiment

```bash
# Run the full 3×3×3 factorial experiment
python scripts/run_experiment.py --mode standard --output-dir results/full
```

## 📊 Experimental Design

### Independent Variables

#### 1. Architecture (Team Topology)

- **Debate+Vote**: Parallel agents debate, judge aggregates
- **Orchestrator→Sub-agents**: Planner/critic/executors with central coordination
- **Role-play Teamwork**: Peer-to-peer collaboration (CAMEL-style)

#### 2. Agent Variance (3 levels)

- **Low**: Same LLM, same prompt template, low temperature
- **Medium**: Same LLM family, different prompts/temperature
- **High**: Different LLMs, different prompting styles

#### 3. Leader Dominance (3 levels)

- **None**: Equal vote (majority or mean of confidences)
- **Moderate**: Leader has tie-break + agenda setting
- **Strong**: Leader has soft-veto or final say

### Tasks & Datasets

- **Math Reasoning**: GSM8K (easy), MATH (hard)
- **Multi-hop QA**: HotpotQA
- **Code Generation**: HumanEval (pass@k)

### Outcome Metrics

#### Task Performance

- Accuracy, Exact Match, F1 Score, Pass@k

#### Process Metrics

- Deliberation cost (tokens, wall-clock time)
- Consensus difficulty (rounds, disagreement rate)
- Judge robustness (aggregation method performance)

#### Diversity/Variance Measures

- Output disagreement (pairwise mismatch, rationale edit distance)
- Distributional distance (KL divergence, embedding dispersion)
- Source heterogeneity (model/prompt/temperature diversity)

## 🔬 Usage Examples

### Basic Experiment

```python
from src.core.experiment import run_experiment
from src.core.config import create_standard_experiment

# Create experiment configuration
config = create_standard_experiment()
config.num_samples_per_condition = 100
config.num_replications = 5

# Run experiment
result = run_experiment(config)
print(f"Experiment completed with {len(result.results)} results")
```

### Custom Configuration

```python
from src.core.config import ExperimentConfig, Architecture, VarianceLevel, DominanceLevel

# Create custom experiment
config = ExperimentConfig(
    architectures=[Architecture.DEBATE_VOTE, Architecture.ORCHESTRATOR_SUBAGENTS],
    variance_levels=[VarianceLevel.LOW, VarianceLevel.HIGH],
    dominance_levels=[DominanceLevel.NONE, DominanceLevel.STRONG],
    tasks=["math_reasoning"],
    datasets=["gsm8k"],
    num_samples_per_condition=50
)

result = run_experiment(config)
```

### Baseline Comparison

```python
from src.baselines import CoTBaseline, SelfConsistencyBaseline, MixtureOfAgentsBaseline
from src.baselines.single_agent import BaselineConfig

# Test different baselines
cot_baseline = CoTBaseline(BaselineConfig())
sc_baseline = SelfConsistencyBaseline(BaselineConfig(num_samples=5))
moa_baseline = MixtureOfAgentsBaseline(MoAConfig(num_agents=4))

# Run on a task
result = await cot_baseline.solve_task(task_item)
```

## 📈 Analysis and Results

The framework provides comprehensive analysis tools:

### Statistical Analysis

- Mixed-effects modeling with random intercepts
- Main effects and interaction analysis
- Post-hoc comparisons and effect sizes

### Key Research Questions

1. **Main Effect**: Does variance level affect team performance?
2. **Main Effect**: Does dominance level affect team performance?
3. **Interaction**: Do high-variance teams benefit more from strong leaders?
4. **Architecture**: Which architectures work best for different tasks?

### Expected Findings

- High-variance teams should benefit more from strong leadership
- Different architectures may excel at different task types
- Process metrics should correlate with performance outcomes

## 🛠️ Development

### Adding New Architectures

```python
from src.core.team import Team, Architecture

class NewArchitectureTeam(Team):
    async def solve_task(self, task, context=None):
        # Implement your architecture
        pass

# Register in team.py
def create_team(config):
    if config.architecture == Architecture.NEW_ARCHITECTURE:
        return NewArchitectureTeam(config)
    # ... existing code
```

### Adding New Tasks

```python
from src.core.task import Task, TaskType

class NewTask(Task):
    def load_dataset(self):
        # Load your dataset
        pass

    def evaluate_response(self, response, ground_truth, metadata=None):
        # Implement evaluation logic
        pass

# Register in task.py
def create_task(task_type, dataset, config=None):
    if task_type == TaskType.NEW_TASK:
        return NewTask(config)
    # ... existing code
```

### Adding New Metrics

```python
from src.evaluation.metrics import MetricsCalculator

class NewMetrics(MetricsCalculator):
    def calculate_metrics(self, team_response, ground_truth, task_metadata=None):
        # Implement your metrics
        pass
```

## 📚 References

- [Debate+Vote Architecture](https://arxiv.org/abs/2305.14325)
- [Orchestrator Architecture](https://arxiv.org/abs/2308.08155)
- [Role-play Teamwork (CAMEL)](https://arxiv.org/abs/2303.17760)
- [Mixture of Agents](https://arxiv.org/abs/2406.04692)
- [GSM8K Dataset](https://arxiv.org/abs/2110.14168)
- [MATH Dataset](https://arxiv.org/abs/2110.14168)
- [HotpotQA Dataset](https://arxiv.org/abs/1809.09600)
- [HumanEval Dataset](https://arxiv.org/abs/2107.03374)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for multi-agent system research
- Inspired by recent work on agent collaboration and team dynamics
- Designed for systematic evaluation of team structures and leadership patterns
