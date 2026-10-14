from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

# --- 1. Evaluator ---

class ModelEvaluator:
    """Evaluate model performance."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray = None) -> dict:
        """Compute evaluation metrics."""
        result = {
            'accuracy': accuracy_score(y_true, y_pred),
            'n_samples': len(y_true),
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(len(y_true) - np.sum(y_true)),
        }

        if y_prob is not None:
            try:
                result['log_loss'] = log_loss(y_true, y_prob)
            except ValueError:
                result['log_loss'] = None

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        result['true_negatives'] = int(cm[0][0])
        result['false_positives'] = int(cm[0][1])
        result['false_negatives'] = int(cm[1][0])
        result['true_positives'] = int(cm[1][1])

        # Precision and recall
        tp = result['true_positives']
        fp = result['false_positives']
        fn = result['false_negatives']

        result['precision'] = tp / max(tp + fp, 1)
        result['recall'] = tp / max(tp + fn, 1)
        result['f1'] = 2 * result['precision'] * result['recall'] / max(result['precision'] + result['recall'], 1e-10)

        return result

    def compare_models(self, results: List[dict]) -> dict:
        """Compare multiple model evaluation results."""
        if not results:
            return {}

        best_accuracy = max(r['accuracy'] for r in results)
        best_f1 = max(r.get('f1', 0) for r in results)

        return {
            'n_models': len(results),
            'best_accuracy': best_accuracy,
            'best_f1': best_f1,
            'mean_accuracy': np.mean([r['accuracy'] for r in results]),
            'mean_f1': np.mean([r.get('f1', 0) for r in results]),
        }


# --- 2. Test it ---

if __name__ == '__main__':
    evaluator = ModelEvaluator()

    # Dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    y_prob = np.array([
        [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2],
        [0.6, 0.4], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.5, 0.5],
    ])

    result = evaluator.evaluate(y_true, y_pred, y_prob)
    print("=== Evaluation ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nDone.")
