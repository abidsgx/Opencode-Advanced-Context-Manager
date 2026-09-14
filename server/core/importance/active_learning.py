import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .model import ImportanceModel, get_model

# --- 1. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
PENDING_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'training' / 'pending_labels.jsonl'
LABELS_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'training' / 'labels.jsonl'


# --- 2. Active learner ---

class ActiveLearner:
    def __init__(self):
        self.model = get_model()
        PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    def identify_uncertain(self, X: np.ndarray, file_paths: list,
                           feature_names: list = None) -> list:
        """Find samples where model is most uncertain.

        Uses margin sampling: picks samples where P(class_0) and P(class_1)
        are closest to each other.
        """
        if not self.model.initialized:
            return []

        proba = self.model.predict_proba(X)
        # Margin = |P(0) - P(1)|, smaller = more uncertain
        margins = np.abs(proba[:, 0] - proba[:, 1])

        # Sort by margin (most uncertain first)
        uncertain_indices = np.argsort(margins)[:10]

        samples = []
        for idx in uncertain_indices:
            samples.append({
                'sample_id': f"sample_{int(time.time())}_{idx}",
                'file_path': file_paths[idx] if idx < len(file_paths) else f"unknown_{idx}",
                'features': X[idx].tolist(),
                'probability': proba[idx].tolist(),
                'margin': float(margins[idx]),
                'timestamp': time.time(),
            })

        return samples

    def get_pending_samples(self) -> dict:
        """Get pending samples that need user labeling."""
        if not PENDING_PATH.exists():
            return {'samples': [], 'count': 0}

        samples = []
        with open(PENDING_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        return {'samples': samples, 'count': len(samples)}

    def submit_labels(self, labels: list) -> dict:
        """Submit user labels and feed to model.

        Each label: {'sample_id': str, 'score': float 0-10}
        """
        labeled = []
        with open(LABELS_PATH, 'a') as f:
            for label in labels:
                entry = {
                    'sample_id': label['sample_id'],
                    'score': label['score'],
                    'timestamp': time.time(),
                }
                f.write(json.dumps(entry) + '\n')
                labeled.append(entry)

        # Remove labeled samples from pending
        self._cleanup_pending([label['sample_id'] for label in labeled])

        # Convert scores to binary labels and update model
        if self.model.initialized and labeled:
            self._update_model_from_labels(labeled)

        return {'status': 'ok', 'labeled': len(labeled)}

    def _cleanup_pending(self, labeled_ids: list):
        """Remove labeled samples from pending file."""
        if not PENDING_PATH.exists():
            return

        remaining = []
        with open(PENDING_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    if sample.get('sample_id') not in labeled_ids:
                        remaining.append(sample)

        with open(PENDING_PATH, 'w') as f:
            for sample in remaining:
                f.write(json.dumps(sample) + '\n')

    def _update_model_from_labels(self, labels: list):
        """Feed labeled data to model with time-decay weights."""
        for label in labels:
            score = label['score']
            # Binary: score >= 5 is "important" (1), else "not important" (0)
            binary_label = 1 if score >= 5.0 else 0

            # We need the features for this sample
            # For now, we store features in pending and retrieve them
            sample = self._find_sample_features(label['sample_id'])
            if sample is not None:
                X = np.array([sample['features']])
                y = np.array([binary_label])

                # Time decay weight (label is fresh, so weight = 1.0)
                weights = np.array([1.0])

                self.model.partial_fit(X, y, sample_weight=weights)
                self.model.save()

    def _find_sample_features(self, sample_id: str):
        """Find sample features from pending or recent predictions."""
        # Check pending file
        if PENDING_PATH.exists():
            with open(PENDING_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        if sample.get('sample_id') == sample_id:
                            return sample
        return None


# --- 3. Test it ---

if __name__ == '__main__':
    np.random.seed(42)

    # Create dummy model
    from .model import train_initial
    X_dummy = np.random.randn(50, 19)
    y_dummy = np.random.randint(0, 2, 50)
    model = train_initial(X_dummy)

    # Create active learner
    learner = ActiveLearner()

    # Find uncertain samples
    X_test = np.random.randn(20, 19)
    files = [f"src/file_{i}.py" for i in range(20)]

    uncertain = learner.identify_uncertain(X_test, files)
    print(f"Found {len(uncertain)} uncertain samples")
    for s in uncertain[:3]:
        print(f"  {s['file_path']}: margin={s['margin']:.4f}, prob={s['probability']}")

    # Test pending
    pending = learner.get_pending_samples()
    print(f"\nPending samples: {pending['count']}")

    print("\nDone.")
