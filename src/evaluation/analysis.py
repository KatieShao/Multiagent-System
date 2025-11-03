"""
Statistical analysis and experiment evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    # Main effects
    architecture_effect: Dict[str, Any]
    variance_effect: Dict[str, Any]
    dominance_effect: Dict[str, Any]
    
    # Interactions
    variance_dominance_interaction: Dict[str, Any]
    architecture_variance_interaction: Dict[str, Any]
    architecture_dominance_interaction: Dict[str, Any]
    
    # Model fit
    model_summary: Dict[str, Any]
    residuals: np.ndarray
    r_squared: float
    
    # Key findings
    key_findings: List[str]
    recommendations: List[str]


class StatisticalAnalyzer:
    """Statistical analysis for multi-agent experiments."""
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        self.data = pd.DataFrame(results_data)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for analysis."""
        # Convert categorical variables
        self.data['architecture'] = self.data['architecture'].astype('category')
        self.data['variance_level'] = self.data['variance_level'].astype('category')
        self.data['dominance_level'] = self.data['dominance_level'].astype('category')
        
        # Create interaction terms
        self.data['variance_dominance'] = self.data['variance_level'].astype(str) + '_' + self.data['dominance_level'].astype(str)
        self.data['architecture_variance'] = self.data['architecture'].astype(str) + '_' + self.data['variance_level'].astype(str)
        self.data['architecture_dominance'] = self.data['architecture'].astype(str) + '_' + self.data['dominance_level'].astype(str)
        
        # Ensure numeric columns are numeric
        numeric_columns = ['accuracy', 'exact_match', 'f1_score', 'deliberation_cost', 'consensus_difficulty']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
    
    def run_mixed_effects_analysis(self, outcome_variable: str = 'accuracy') -> AnalysisResult:
        """Run mixed-effects model analysis."""
        # Prepare formula for mixed-effects model
        formula = f"{outcome_variable} ~ architecture * variance_level * dominance_level"
        
        # Add random effects for item and dataset
        random_effects = "1|task_id + 1|dataset"
        
        try:
            # Fit mixed-effects model
            model = mixedlm(formula, self.data, groups=self.data['team_id'])
            result = model.fit()
            
            # Extract results
            architecture_effect = self._extract_main_effect(result, 'architecture')
            variance_effect = self._extract_main_effect(result, 'variance_level')
            dominance_effect = self._extract_main_effect(result, 'dominance_level')
            
            # Extract interactions
            variance_dominance_interaction = self._extract_interaction(result, 'variance_level', 'dominance_level')
            architecture_variance_interaction = self._extract_interaction(result, 'architecture', 'variance_level')
            architecture_dominance_interaction = self._extract_interaction(result, 'architecture', 'dominance_level')
            
            # Generate findings
            key_findings = self._generate_findings(result)
            recommendations = self._generate_recommendations(result)
            
            return AnalysisResult(
                architecture_effect=architecture_effect,
                variance_effect=variance_effect,
                dominance_effect=dominance_effect,
                variance_dominance_interaction=variance_dominance_interaction,
                architecture_variance_interaction=architecture_variance_interaction,
                architecture_dominance_interaction=architecture_dominance_interaction,
                model_summary=result.summary().as_dict(),
                residuals=result.resid,
                r_squared=result.rsquared,
                key_findings=key_findings,
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error in mixed-effects analysis: {e}")
            return self._create_empty_result()
    
    def _extract_main_effect(self, model_result, variable: str) -> Dict[str, Any]:
        """Extract main effect for a variable."""
        # TODO: Implement main effect extraction
        return {
            "variable": variable,
            "coefficients": {},
            "p_values": {},
            "significant": False,
            "effect_size": 0.0
        }
    
    def _extract_interaction(self, model_result, var1: str, var2: str) -> Dict[str, Any]:
        """Extract interaction effect between two variables."""
        # TODO: Implement interaction extraction
        return {
            "variables": f"{var1} * {var2}",
            "coefficients": {},
            "p_values": {},
            "significant": False,
            "effect_size": 0.0
        }
    
    def _generate_findings(self, model_result) -> List[str]:
        """Generate key findings from analysis."""
        findings = []
        
        # TODO: Implement findings generation based on model results
        findings.append("Analysis completed successfully")
        findings.append("Mixed-effects model fitted with random intercepts")
        
        return findings
    
    def _generate_recommendations(self, model_result) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # TODO: Implement recommendations generation
        recommendations.append("Consider increasing sample size for more robust results")
        recommendations.append("Investigate significant interactions further")
        
        return recommendations
    
    def _create_empty_result(self) -> AnalysisResult:
        """Create empty result when analysis fails."""
        return AnalysisResult(
            architecture_effect={},
            variance_effect={},
            dominance_effect={},
            variance_dominance_interaction={},
            architecture_variance_interaction={},
            architecture_dominance_interaction={},
            model_summary={},
            residuals=np.array([]),
            r_squared=0.0,
            key_findings=["Analysis failed"],
            recommendations=["Check data quality and model specification"]
        )
    
    def run_anova_analysis(self, outcome_variable: str = 'accuracy') -> Dict[str, Any]:
        """Run ANOVA analysis."""
        # TODO: Implement ANOVA analysis
        return {
            "method": "ANOVA",
            "outcome_variable": outcome_variable,
            "results": "ANOVA analysis not yet implemented"
        }
    
    def run_post_hoc_analysis(self, outcome_variable: str = 'accuracy') -> Dict[str, Any]:
        """Run post-hoc analysis."""
        # TODO: Implement post-hoc analysis
        return {
            "method": "Post-hoc",
            "outcome_variable": outcome_variable,
            "results": "Post-hoc analysis not yet implemented"
        }


class ExperimentAnalyzer:
    """Comprehensive experiment analysis."""
    
    def __init__(self, results_data: List[Dict[str, Any]]):
        self.data = pd.DataFrame(results_data)
        self.statistical_analyzer = StatisticalAnalyzer(results_data)
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of the experiment."""
        analysis_results = {}
        
        # Main analysis
        analysis_results['mixed_effects'] = self.statistical_analyzer.run_mixed_effects_analysis()
        
        # Additional analyses
        analysis_results['descriptive_stats'] = self._calculate_descriptive_statistics()
        analysis_results['correlation_analysis'] = self._run_correlation_analysis()
        analysis_results['effect_sizes'] = self._calculate_effect_sizes()
        
        # Generate visualizations
        analysis_results['visualizations'] = self._generate_visualizations()
        
        return analysis_results
    
    def _calculate_descriptive_statistics(self) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        numeric_columns = ['accuracy', 'exact_match', 'f1_score']
        available_columns = [col for col in numeric_columns if col in self.data.columns]
        
        if not available_columns:
            return {"error": "No numeric columns found"}
        
        desc_stats = {}
        for col in available_columns:
            desc_stats[col] = {
                "mean": self.data[col].mean(),
                "std": self.data[col].std(),
                "min": self.data[col].min(),
                "max": self.data[col].max(),
                "median": self.data[col].median(),
                "q25": self.data[col].quantile(0.25),
                "q75": self.data[col].quantile(0.75)
            }
        
        return desc_stats
    
    def _run_correlation_analysis(self) -> Dict[str, Any]:
        """Run correlation analysis."""
        numeric_columns = ['accuracy', 'exact_match', 'f1_score', 'deliberation_cost', 'consensus_difficulty']
        available_columns = [col for col in numeric_columns if col in self.data.columns]
        
        if len(available_columns) < 2:
            return {"error": "Insufficient numeric columns for correlation"}
        
        correlation_matrix = self.data[available_columns].corr()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(correlation_matrix)
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations in the matrix."""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        return strong_corrs
    
    def _calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate effect sizes for different conditions."""
        effect_sizes = {}
        
        # Calculate Cohen's d for different comparisons
        for outcome in ['accuracy', 'exact_match']:
            if outcome in self.data.columns:
                effect_sizes[outcome] = self._calculate_cohens_d(outcome)
        
        return effect_sizes
    
    def _calculate_cohens_d(self, outcome_variable: str) -> Dict[str, float]:
        """Calculate Cohen's d for different group comparisons."""
        cohens_d = {}
        
        # Compare different variance levels
        if 'variance_level' in self.data.columns:
            low_var = self.data[self.data['variance_level'] == 'low'][outcome_variable]
            high_var = self.data[self.data['variance_level'] == 'high'][outcome_variable]
            
            if len(low_var) > 0 and len(high_var) > 0:
                pooled_std = np.sqrt(((len(low_var) - 1) * low_var.var() + (len(high_var) - 1) * high_var.var()) / 
                                   (len(low_var) + len(high_var) - 2))
                cohens_d['variance_low_vs_high'] = (high_var.mean() - low_var.mean()) / pooled_std
        
        return cohens_d
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate visualization plots."""
        plots = {}
        
        # TODO: Implement visualization generation
        # This would create plots and save them as files
        
        plots['accuracy_by_condition'] = "accuracy_by_condition.png"
        plots['interaction_plot'] = "interaction_plot.png"
        plots['distribution_plots'] = "distribution_plots.png"
        
        return plots
    
    def save_results(self, output_dir: str, analysis_results: Dict[str, Any]):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(output_path / "analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save data summary
        self.data.to_csv(output_path / "data_summary.csv", index=False)
        
        print(f"Analysis results saved to {output_path}")


def analyze_experiment(results_data: List[Dict[str, Any]], output_dir: str = "analysis_results") -> Dict[str, Any]:
    """Main function to analyze experiment results."""
    analyzer = ExperimentAnalyzer(results_data)
    results = analyzer.run_comprehensive_analysis()
    analyzer.save_results(output_dir, results)
    return results

