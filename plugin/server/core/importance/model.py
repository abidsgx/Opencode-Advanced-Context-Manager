import json
import os
import time
from collections import deque
from pathlib import Path

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
ml_config = config.get('importance', {}).get('ml', {})
MODEL_PATH = ml_config.get('model_path', 'data/models/importance_model.pkl')
SCALER_PATH = ml_config.get('scaler_path', 'data/models/feature_scaler.pkl')
CONFIDENCE_THRESHOLD = ml_config.get('confidence_threshold', 0.6)
WINDOW_SIZE = ml_config.get('window_size', 50)
HALF_LIFE = ml_config.get('half_life', 30)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# --- 2. Model class ---

class ImportanceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.initialized = False

    def initialize(self, X_train: np.ndarray):
        """Initial fit on training data."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.001,
            learning_rate='adaptive',
            random_state=42
        )
        # Use partial_fit with both classes to initialize
        self.model.partial_fit(X_scaled, np.zeros(len(X_scaled)), classes=[0, 1])

        self.save()
        self.initialized = True

    def load(self):
        """Load saved model and scaler."""
        model_path = PROJECT_ROOT / MODEL_PATH
        scaler_path = PROJECT_ROOT / SCALER_PATH

        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.initialized = True
            print("Model and scaler loaded.")
        else:
            print("No saved model found. Call initialize() first.")

    def save(self):
        """Save model and scaler to disk."""
        model_path = PROJECT_ROOT / MODEL_PATH
        scaler_path = PROJECT_ROOT / SCALER_PATH

        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print("Model and scaler saved.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.initialized:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.initialized:
            return np.zeros((len(X), 2))
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Incremental update."""
        if not self.initialized:
            self.initialize(X)
            return
        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y, sample_weight=sample_weight, classes=[0, 1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.initialized:
            return 0.0
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)

    def is_uncertain(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask where prediction confidence is below threshold."""
        if not self.initialized:
            return np.ones(len(X), dtype=bool)
        proba = self.predict_proba(X)
        max_proba = np.max(proba, axis=1)
        return max_proba < CONFIDENCE_THRESHOLD


# --- 3. Streaming trainer ---

class StreamingTrainer:
    def __init__(self, model: ImportanceModel, window_size: int = WINDOW_SIZE, half_life: int = HALF_LIFE):
        self.model = model
        self.window_size = window_size
        self.half_life = half_life

        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)

        self.predictions = []
        self.true_vals = []
        self.probabilities = []

        self.train_sizes = []
        self.train_scores = []
        self.test_scores = []
        self.loss_history = []
        self.iterations = []
        self.current_size = 0
        self.iteration_counter = 0

    def process_point(self, x: np.ndarray, y_true: int):
        """Predict, buffer, then update when buffer is full."""
        x_point = x.reshape(1, -1)

        # 1. Predict before updating
        pred = self.model.predict(x_point)[0]
        prob = self.model.predict_proba(x_point)[0][1]

        self.predictions.append(pred)
        self.true_vals.append(y_true)
        self.probabilities.append(prob)

        # 2. Buffer the point
        self.x_buffer.append(x)
        self.y_buffer.append(y_true)

        # 3. Update model when buffer is full
        if len(self.x_buffer) == self.window_size:
            X_batch = np.array(self.x_buffer)
            y_batch = np.array(self.y_buffer)

            # Time decay weights (same as reference script)
            time_diffs = np.arange(self.window_size - 1, -1, -1)
            weights = np.exp(-(np.log(2) * time_diffs) / self.half_life)

            self.model.partial_fit(X_batch, y_batch, sample_weight=weights)

            # Track metrics
            self.current_size += len(X_batch)
            self.train_sizes.append(self.current_size)
            self.train_scores.append(self.model.score(X_batch, y_batch))

    def get_accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        return accuracy_score(self.true_vals, self.predictions)

    def get_log_loss(self) -> float:
        if not self.probabilities:
            return 0.0
        try:
            return log_loss(self.true_vals, self.probabilities)
        except ValueError:
            return 0.0


# --- 4. High-level interface ---

_model = ImportanceModel()

def get_model() -> ImportanceModel:
    global _model
    if not _model.initialized:
        _model.load()
    return _model


def train_initial(X_train: np.ndarray):
    """Initial training on labeled data."""
    model = get_model()
    model.initialize(X_train)
    return model


def online_update(X_batch: np.ndarray, y_batch: np.ndarray):
    """Incremental update with time-decay weights."""
    model = get_model()
    window_size = len(X_batch)
    time_diffs = np.arange(window_size - 1, -1, -1)
    weights = np.exp(-(np.log(2) * time_diffs) / HALF_LIFE)
    model.partial_fit(X_batch, y_batch, sample_weight=weights)


def predict_importance(X: np.ndarray) -> dict:
    """Predict importance scores and confidence."""
    model = get_model()
    proba = model.predict_proba(X)
    predictions = model.predict(X)
    uncertain = model.is_uncertain(X)

    return {
        'predictions': predictions.tolist(),
        'probabilities': proba.tolist(),
        'uncertain': uncertain.tolist(),
    }


# --- 5. Test it ---

if __name__ == '__main__':
    # Generate dummy data
    np.random.seed(42)
    n_samples = 200
    n_features = 19

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 3] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initial train
    print("=== Initial Training ===")
    model = train_initial(X_train)
    print(f"Initial accuracy: {model.score(X_train, np.zeros(len(X_train))):.4f}")

    # Streaming
    print("\n=== Streaming Update ===")
    trainer = StreamingTrainer(model, window_size=20, half_life=10)

    for i in range(len(X_test)):
        trainer.process_point(X_test[i], y_test[i])

    print(f"Streaming accuracy: {trainer.get_accuracy():.4f}")
    print(f"Streaming log-loss: {trainer.get_log_loss():.4f}")

    # Predictions
    result = predict_importance(X_test[:5])
    print(f"\nSample predictions: {result['predictions']}")
    print(f"Sample probabilities: {[f'{p[1]:.3f}' for p in result['probabilities']]}")
    print(f"Uncertain: {result['uncertain']}")

    print("\nDone.")
