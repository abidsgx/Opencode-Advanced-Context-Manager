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


config = load_config()
TRAINING_DIR = Path(__file__).parent.parent.parent / 'data' / 'training'


# --- 2. Data collector ---

class DataCollector:
    """Collect training data from session edits and active learning."""

    def __init__(self):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        self.data_path = TRAINING_DIR / 'training_data.jsonl'

    def record_edit(self, file_path: str, function_name: str, features: np.ndarray,
                    session_id: str, label: int = None):
        """Record an edit with features."""
        entry = {
            'file_path': file_path,
            'function_name': function_name,
            'features': features.tolist(),
            'session_id': session_id,
            'timestamp': time.time(),
            'label': label,
        }
        with open(self.data_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all labeled data."""
        if not self.data_path.exists():
            return np.array([]), np.array([])

        X_list, y_list = [], []
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get('label') is not None:
                    X_list.append(entry['features'])
                    y_list.append(entry['label'])

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def get_unlabeled_data(self) -> list:
        """Get entries without labels."""
        if not self.data_path.exists():
            return []

        entries = []
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get('label') is None:
                    entries.append(entry)

        return entries

    def apply_labels(self, labels: dict):
        """Apply labels to unlabeled data."""
        if not self.data_path.exists():
            return

        entries = []
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get('label') is None and entry['file_path'] in labels:
                    entry['label'] = labels[entry['file_path']]
                entries.append(entry)

        with open(self.data_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        if not self.data_path.exists():
            return {'total': 0, 'labeled': 0, 'unlabeled': 0}

        total, labeled, unlabeled = 0, 0, 0
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                entry = json.loads(line)
                if entry.get('label') is not None:
                    labeled += 1
                else:
                    unlabeled += 1

        return {'total': total, 'labeled': labeled, 'unlabeled': unlabeled}


# --- 3. Synthetic data generator ---

class SyntheticGenerator:
    """Generate synthetic training data from code metrics."""

    def generate_from_file_metrics(self, file_metrics: dict, n_samples: int = 100) -> list:
        """Generate synthetic labeled samples from file metrics."""
        samples = []

        for i in range(n_samples):
            # Random file features
            features = np.random.randn(19)

            # Simple rules for labeling
            # Files with high edit frequency and centrality are likely important
            edit_freq = features[2]  # edit_frequency
            centrality = features[8]  # degree_centrality
            goal_align = features[6]  # functions_modified (proxy for goal alignment)

            score = 0.3 * edit_freq + 0.4 * centrality + 0.3 * goal_align
            label = 1 if score > 0 else 0

            samples.append({
                'features': features.tolist(),
                'label': label,
                'source': 'synthetic',
            })

        return samples

    def augment_real_data(self, X: np.ndarray, y: np.ndarray, n_augment: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Augment real data with noise."""
        augmented_X = []
        augmented_y = []

        for _ in range(n_augment):
            idx = np.random.randint(len(X))
            x = X[idx]
            noise = np.random.randn(len(x)) * 0.1
            augmented_X.append(x + noise)
            augmented_y.append(y[idx])

        if augmented_X:
            X_aug = np.array(augmented_X)
            y_aug = np.array(augmented_y)
            return np.vstack([X, X_aug]), np.concatenate([y, y_aug])

        return X, y


# --- 4. Test it ---

if __name__ == '__main__':
    collector = DataCollector()
    generator = SyntheticGenerator()

    # Generate synthetic data
    synthetic = generator.generate_from_file_metrics({}, n_samples=50)
    print(f"Generated {len(synthetic)} synthetic samples")
    print(f"  Label distribution: {sum(1 for s in synthetic if s['label'] == 1)} positive, "
          f"{sum(1 for s in synthetic if s['label'] == 0)} negative")

    # Record some edits
    for i in range(10):
        features = np.random.randn(19)
        collector.record_edit(f"src/file_{i}.py", f"func_{i}", features, "session_1")

    # Stats
    stats = collector.get_stats()
    print(f"\nDataset stats: {stats}")

    # Get unlabeled
    unlabeled = collector.get_unlabeled_data()
    print(f"Unlabeled samples: {len(unlabeled)}")

    print("\nDone.")
