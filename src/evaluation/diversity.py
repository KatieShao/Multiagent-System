"""
Diversity and variance calculation for multi-agent systems.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


@dataclass
class DiversityMetrics:
    """Container for diversity metrics."""
    # Output disagreement
    pairwise_disagreement: float
    consensus_score: float
    rationale_edit_distance: float
    
    # Distributional distance
    kl_divergence: float
    embedding_dispersion: float
    response_variance: float
    
    # Source heterogeneity
    model_diversity: float
    prompt_diversity: float
    temperature_diversity: float
    overall_heterogeneity: float


class DiversityCalculator:
    """Calculate diversity and variance metrics for multi-agent teams."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def calculate_diversity_metrics(
        self, 
        individual_responses: Dict[str, Dict[str, Any]],
        agent_configs: List[Dict[str, Any]]
    ) -> DiversityMetrics:
        """Calculate comprehensive diversity metrics."""
        
        # Extract response texts and metadata
        response_texts = [resp.get("text", "") for resp in individual_responses.values()]
        response_metadata = [resp.get("metadata", {}) for resp in individual_responses.values()]
        
        # Calculate output disagreement metrics
        output_disagreement = self._calculate_output_disagreement(response_texts)
        
        # Calculate distributional distance metrics
        distributional_distance = self._calculate_distributional_distance(response_texts)
        
        # Calculate source heterogeneity metrics
        source_heterogeneity = self._calculate_source_heterogeneity(agent_configs, response_metadata)
        
        return DiversityMetrics(
            pairwise_disagreement=output_disagreement["pairwise_disagreement"],
            consensus_score=output_disagreement["consensus_score"],
            rationale_edit_distance=output_disagreement["rationale_edit_distance"],
            kl_divergence=distributional_distance["kl_divergence"],
            embedding_dispersion=distributional_distance["embedding_dispersion"],
            response_variance=distributional_distance["response_variance"],
            model_diversity=source_heterogeneity["model_diversity"],
            prompt_diversity=source_heterogeneity["prompt_diversity"],
            temperature_diversity=source_heterogeneity["temperature_diversity"],
            overall_heterogeneity=source_heterogeneity["overall_heterogeneity"]
        )
    
    def _calculate_output_disagreement(self, response_texts: List[str]) -> Dict[str, float]:
        """Calculate output disagreement metrics."""
        if len(response_texts) < 2:
            return {
                "pairwise_disagreement": 0.0,
                "consensus_score": 1.0,
                "rationale_edit_distance": 0.0
            }
        
        # Calculate pairwise disagreement
        disagreements = []
        for i in range(len(response_texts)):
            for j in range(i + 1, len(response_texts)):
                disagreement = self._calculate_text_disagreement(response_texts[i], response_texts[j])
                disagreements.append(disagreement)
        
        pairwise_disagreement = np.mean(disagreements) if disagreements else 0.0
        consensus_score = 1.0 - pairwise_disagreement
        
        # Calculate rationale edit distance
        rationale_edit_distance = self._calculate_rationale_edit_distance(response_texts)
        
        return {
            "pairwise_disagreement": pairwise_disagreement,
            "consensus_score": consensus_score,
            "rationale_edit_distance": rationale_edit_distance
        }
    
    def _calculate_text_disagreement(self, text1: str, text2: str) -> float:
        """Calculate disagreement between two text responses."""
        # Tokenize and normalize
        tokens1 = set(self._tokenize_text(text1))
        tokens2 = set(self._tokenize_text(text2))
        
        if not tokens1 and not tokens2:
            return 0.0
        if not tokens1 or not tokens2:
            return 1.0
        
        # Calculate Jaccard distance
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for comparison."""
        # Simple tokenization - can be enhanced with more sophisticated methods
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _calculate_rationale_edit_distance(self, response_texts: List[str]) -> float:
        """Calculate edit distance between rationales."""
        if len(response_texts) < 2:
            return 0.0
        
        # Extract rationales (assume they're in the text)
        rationales = [self._extract_rationale(text) for text in response_texts]
        
        # Calculate pairwise edit distances
        edit_distances = []
        for i in range(len(rationales)):
            for j in range(i + 1, len(rationales)):
                distance = self._levenshtein_distance(rationales[i], rationales[j])
                max_len = max(len(rationales[i]), len(rationales[j]))
                normalized_distance = distance / max_len if max_len > 0 else 0.0
                edit_distances.append(normalized_distance)
        
        return np.mean(edit_distances) if edit_distances else 0.0
    
    def _extract_rationale(self, text: str) -> str:
        """Extract rationale from response text."""
        # Simple extraction - look for reasoning indicators
        lines = text.split('\n')
        rationale_lines = []
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in ['because', 'since', 'therefore', 'thus', 'so']):
                rationale_lines.append(line)
        
        return ' '.join(rationale_lines) if rationale_lines else text
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_distributional_distance(self, response_texts: List[str]) -> Dict[str, float]:
        """Calculate distributional distance metrics."""
        if len(response_texts) < 2:
            return {
                "kl_divergence": 0.0,
                "embedding_dispersion": 0.0,
                "response_variance": 0.0
            }
        
        # Calculate KL divergence
        kl_divergence = self._calculate_kl_divergence(response_texts)
        
        # Calculate embedding dispersion
        embedding_dispersion = self._calculate_embedding_dispersion(response_texts)
        
        # Calculate response variance
        response_variance = self._calculate_response_variance(response_texts)
        
        return {
            "kl_divergence": kl_divergence,
            "embedding_dispersion": embedding_dispersion,
            "response_variance": response_variance
        }
    
    def _calculate_kl_divergence(self, response_texts: List[str]) -> float:
        """Calculate KL divergence between response distributions."""
        # Create vocabulary from all responses
        all_tokens = []
        for text in response_texts:
            all_tokens.extend(self._tokenize_text(text))
        
        vocabulary = list(set(all_tokens))
        if not vocabulary:
            return 0.0
        
        # Calculate token distributions for each response
        distributions = []
        for text in response_texts:
            tokens = self._tokenize_text(text)
            token_counts = Counter(tokens)
            
            # Create probability distribution
            total_tokens = len(tokens)
            distribution = [token_counts.get(token, 0) / total_tokens for token in vocabulary]
            distributions.append(distribution)
        
        # Calculate pairwise KL divergences
        kl_divergences = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                kl_div = self._kl_divergence(distributions[i], distributions[j])
                kl_divergences.append(kl_div)
        
        return np.mean(kl_divergences) if kl_divergences else 0.0
    
    def _kl_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate KL divergence between two distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = np.array(p) + epsilon
        q = np.array(q) + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
    
    def _calculate_embedding_dispersion(self, response_texts: List[str]) -> float:
        """Calculate embedding dispersion using TF-IDF."""
        if len(response_texts) < 2:
            return 0.0
        
        try:
            # Create TF-IDF embeddings
            tfidf_matrix = self.vectorizer.fit_transform(response_texts)
            
            # Calculate pairwise cosine similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate dispersion as 1 - mean similarity
            # Exclude diagonal (self-similarity)
            mask = np.ones_like(similarity_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            
            mean_similarity = np.mean(similarity_matrix[mask])
            dispersion = 1.0 - mean_similarity
            
            return dispersion
            
        except Exception as e:
            print(f"Error calculating embedding dispersion: {e}")
            return 0.0
    
    def _calculate_response_variance(self, response_texts: List[str]) -> float:
        """Calculate variance in response characteristics."""
        if len(response_texts) < 2:
            return 0.0
        
        # Calculate various response characteristics
        lengths = [len(text) for text in response_texts]
        word_counts = [len(self._tokenize_text(text)) for text in response_texts]
        
        # Calculate coefficient of variation
        length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0.0
        word_cv = np.std(word_counts) / np.mean(word_counts) if np.mean(word_counts) > 0 else 0.0
        
        # Average coefficient of variation
        response_variance = (length_cv + word_cv) / 2
        
        return response_variance
    
    def _calculate_source_heterogeneity(self, agent_configs: List[Dict[str, Any]], response_metadata: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate source heterogeneity metrics."""
        if not agent_configs:
            return {
                "model_diversity": 0.0,
                "prompt_diversity": 0.0,
                "temperature_diversity": 0.0,
                "overall_heterogeneity": 0.0
            }
        
        # Extract agent characteristics
        models = [config.get("model_name", "unknown") for config in agent_configs]
        prompts = [config.get("prompt_template", "unknown") for config in agent_configs]
        temperatures = [config.get("temperature", 0.0) for config in agent_configs]
        
        # Calculate diversity for each characteristic
        model_diversity = self._calculate_categorical_diversity(models)
        prompt_diversity = self._calculate_categorical_diversity(prompts)
        temperature_diversity = self._calculate_numerical_diversity(temperatures)
        
        # Calculate overall heterogeneity
        overall_heterogeneity = (model_diversity + prompt_diversity + temperature_diversity) / 3
        
        return {
            "model_diversity": model_diversity,
            "prompt_diversity": prompt_diversity,
            "temperature_diversity": temperature_diversity,
            "overall_heterogeneity": overall_heterogeneity
        }
    
    def _calculate_categorical_diversity(self, categories: List[str]) -> float:
        """Calculate diversity for categorical variables."""
        if not categories:
            return 0.0
        
        unique_categories = set(categories)
        total_categories = len(categories)
        
        # Calculate Shannon entropy
        category_counts = Counter(categories)
        probabilities = [count / total_categories for count in category_counts.values()]
        
        shannon_entropy = entropy(probabilities)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(unique_categories)) if len(unique_categories) > 1 else 1.0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _calculate_numerical_diversity(self, values: List[float]) -> float:
        """Calculate diversity for numerical variables."""
        if not values or len(values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        cv = std_val / mean_val if mean_val != 0 else 0.0
        
        # Normalize to [0, 1] range (assuming CV > 1 is very high diversity)
        normalized_cv = min(cv, 1.0)
        
        return normalized_cv

