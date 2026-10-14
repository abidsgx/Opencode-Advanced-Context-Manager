import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# --- 1. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
ol_config = config.get('importance', {}).get('ml', {}).get('online_learning', {})
DECAY_RATE = ol_config.get('decay_rate', 0.023)  # Forget 50% every 30 days
MIN_SAMPLES = ol_config.get('min_samples_for_training', 10)
WINDOW_SIZE = ol_config.get('window_size', 50)

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / 'data' / 'training'


# --- 2. Online learner with exponential time decay ---

class OnlineLearner:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.initialized = False

    def load_or_init(self, n_features: int = 19):
        """Load existing model or create new one."""
        model_path = PROJECT_ROOT / 'data' / 'models' / 'online_model.pkl'
        scaler_path = PROJECT_ROOT / 'data' / 'models' / 'online_scaler.pkl'

        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.initialized = True
            print("Online model loaded.")
        else:
            self.scaler = StandardScaler()
            self.model = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                alpha=0.001,
                learning_rate='adaptive',
                random_state=42
            )
            self.initialized = False
            print("New online model created.")

    def initial_fit(self, X: np.ndarray, y: np.ndarray):
        """First fit on labeled data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.initialized = True
        self.save()
        print(f"Initial fit done. Accuracy: {self.model.score(X_scaled, y):.4f}")

    def add_sample(self, x: np.ndarray, label: int, timestamp: float = None):
        """Add a labeled sample to the buffer."""
        timestamp = timestamp or time.time()
        self.buffer.append({
            'x': x,
            'y': label,
            'timestamp': timestamp,
        })

    def update(self):
        """Update model from buffer with time-decay weights."""
        if len(self.buffer) < MIN_SAMPLES:
            return

        if not self.initialized:
            X = np.array([s['x'] for s in self.buffer])
            y = np.array([s['y'] for s in self.buffer])
            self.initial_fit(X, y)
            self.buffer.clear()
            return

        # Build batch
        X_batch = np.array([s['x'] for s in self.buffer])
        y_batch = np.array([s['y'] for s in self.buffer])
        timestamps = np.array([s['timestamp'] for s in self.buffer])

        # Time decay: newer samples have higher weight
        now = time.time()
        age_days = (now - timestamps) / 86400.0
        weights = np.exp(-DECAY_RATE * age_days)

        X_scaled = self.scaler.transform(X_batch)
        self.model.partial_fit(X_scaled, y_batch, sample_weight=weights, classes=[0, 1])

        self.save()
        self.buffer.clear()
        print("Model updated. Buffer cleared.")

    def predict(self, X: np.ndarray) -> dict:
        """Predict with confidence."""
        if not self.initialized:
            return {
                'predictions': [0] * len(X),
                'probabilities': [[0.5, 0.5]] * len(X),
                'uncertain': [True] * len(X),
            }

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        max_proba = np.max(probabilities, axis=1)
        uncertain = max_proba < 0.6

        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'uncertain': uncertain.tolist(),
        }

    def save(self):
        """Save model to disk."""
        model_path = PROJECT_ROOT / 'data' / 'models' / 'online_model.pkl'
        scaler_path = PROJECT_ROOT / 'data' / 'models' / 'online_scaler.pkl'

        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def get_stats(self) -> dict:
        """Get training stats."""
        return {
            'initialized': self.initialized,
            'buffer_size': len(self.buffer),
            'window_size': WINDOW_SIZE,
            'decay_rate': DECAY_RATE,
            'min_samples': MIN_SAMPLES,
        }


# --- 3. Training data collector ---

class DataCollector:
    """Collect training data from session edits."""

    def __init__(self):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        self.data_path = TRAINING_DIR / 'training_data.jsonl'

    def record_edit(self, file_path: str, function_name: str, features: np.ndarray,
                    session_id: str, label: int = None):
        """Record an edit with features for future training."""
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

    def get_labeled_data(self) -> tuple:
        """Get all labeled data as X, y arrays."""
        if not self.data_path.exists():
            return np.array([]), np.array([])

        X_list = []
        y_list = []

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
        """Apply labels to unlabeled data.

        labels: dict mapping file_path -> int label (0 or 1)
        """
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


# --- 4. Test it ---

if __name__ == '__main__':
    np.random.seed(42)

    # Create learner
    learner = OnlineLearner()
    learner.load_or_init(n_features=19)

    # Dummy initial data
    X_init = np.random.randn(30, 19)
    y_init = np.random.randint(0, 2, 30)
    learner.initial_fit(X_init, y_init)

    # Simulate streaming
    print("\n=== Streaming ===")
    collector = DataCollector()

    for i in range(40):
        x = np.random.randn(19)
        label = 1 if x[0] + x[3] > 0 else 0
        learner.add_sample(x, label, timestamp=time.time() - (40 - i) * 60)
        collector.record_edit(f"src/file_{i}.py", f"func_{i}", x, "session_1")

        if i % 10 == 0:
            learner.update()
            print(f"  Step {i}: buffer={len(learner.buffer)}, stats={learner.get_stats()}")

    # Final update
    learner.update()

    # Predict
    X_test = np.random.randn(5, 19)
    result = learner.predict(X_test)
    print(f"\nPredictions: {result['predictions']}")
    print(f"Uncertain: {result['uncertain']}")

    # Collector stats
    X_labeled, y_labeled = collector.get_labeled_data()
    print(f"\nLabeled data: {len(X_labeled)} samples")
    unlabeled = collector.get_unlabeled_data()
    print(f"Unlabeled data: {len(unlabeled)} samples")

    print("\nDone.")
