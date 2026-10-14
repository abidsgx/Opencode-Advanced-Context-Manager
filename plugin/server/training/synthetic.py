import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# --- 1. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# --- 2. Synthetic data generator ---

class SyntheticGenerator:
    """Generate synthetic training data from code metrics and rules."""

    def __init__(self):
        self.config = load_config()

    def generate_from_rules(self, n_samples: int = 200) -> list:
        """Generate synthetic data based on importance rules.

        Rules:
        - Files with high edit frequency and centrality -> important (1)
        - Files with low edit frequency and centrality -> not important (0)
        - Files with high goal alignment -> important (1)
        """
        samples = []

        for _ in range(n_samples):
            # Random features
            features = np.random.randn(19)

            # Compute rule-based label
            edit_freq = features[2]   # edit_frequency
            centrality = features[8]  # degree_centrality
            goal_align = features[6]  # functions_modified (proxy)
            recency = features[3]     # recency_score

            # Weighted rule
            score = (0.3 * edit_freq + 0.3 * centrality +
                     0.2 * goal_align + 0.2 * recency)

            # Add some noise
            score += np.random.randn() * 0.2

            label = 1 if score > 0 else 0
            confidence = min(abs(score) / 2.0, 1.0)

            samples.append({
                'features': features.tolist(),
                'label': label,
                'confidence': confidence,
                'source': 'synthetic_rules',
            })

        return samples

    def generate_from_session(self, session_data: dict, graph_data: dict) -> list:
        """Generate training data from a real session."""
        from features.base import build_all_features, compute_session_metrics

        features = build_all_features(session_data, graph_data)
        metrics = compute_session_metrics(session_data)

        samples = []
        for file_path, feat_vec in features.items():
            # Heuristic label based on metrics
            edits = metrics.edits_per_file.get(file_path, 0)
            centrality = graph_data.get('degree_centrality', {}).get(file_path, 0.0)

            # Simple heuristic: high edits + high centrality = important
            score = np.log1p(edits) * 0.5 + centrality * 0.5
            label = 1 if score > 0.5 else 0

            samples.append({
                'file_path': file_path,
                'features': feat_vec.tolist() if isinstance(feat_vec, np.ndarray) else feat_vec,
                'label': label,
                'source': 'session_derived',
            })

        return samples

    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset by oversampling minority class."""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return X, y

        # Oversample minority
        if len(pos_idx) < len(neg_idx):
            oversample = np.random.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
            X = np.vstack([X, X[oversample]])
            y = np.concatenate([y, y[oversample]])
        elif len(neg_idx) < len(pos_idx):
            oversample = np.random.choice(neg_idx, size=len(pos_idx) - len(neg_idx), replace=True)
            X = np.vstack([X, X[oversample]])
            y = np.concatenate([y, y[oversample]])

        # Shuffle
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


# --- 3. Test it ---

if __name__ == '__main__':
    gen = SyntheticGenerator()

    # Generate from rules
    samples = gen.generate_from_rules(100)
    print(f"Generated {len(samples)} synthetic samples")
    print(f"  Positive: {sum(1 for s in samples if s['label'] == 1)}")
    print(f"  Negative: {sum(1 for s in samples if s['label'] == 0)}")

    # Convert to arrays
    X = np.array([s['features'] for s in samples])
    y = np.array([s['label'] for s in samples])

    # Balance
    X_bal, y_bal = gen.balance_dataset(X, y)
    print(f"\nAfter balancing: {len(X_bal)} samples")
    print(f"  Positive: {np.sum(y_bal == 1)}")
    print(f"  Negative: {np.sum(y_bal == 0)}")

    print("\nDone.")
