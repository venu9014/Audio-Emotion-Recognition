"""
EMOTIVA — Model evaluation module.
Evaluates trained CNN and CNN-LSTM models on the test set.
Generates confusion matrices, classification reports, and comparison CSVs.
"""
import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import load_ravdess_metadata, actor_wise_split
from src.feature_extraction import prepare_dataset


def evaluate_model(model_path: str, X_test: np.ndarray,
                   y_test_encoded: np.ndarray, label_encoder) -> dict:
    """
    Evaluate a single model. Returns metrics dict or None if model not found.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    model = tf.keras.models.load_model(model_path)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test_encoded, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(y_test_encoded, y_pred)

    # Map encoded labels to emotion names for the report
    class_names = [config.EMOTION_LABELS.get(c, str(c)) for c in label_encoder.classes_]
    report = classification_report(
        y_test_encoded, y_pred,
        target_names=class_names,
        zero_division=0
    )

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'report': report,
        'class_names': class_names
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate EMOTIVA SER Models")
    parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Load label encoder
    le_path = os.path.join(config.MODELS_DIR, 'label_encoder.pkl')
    if not os.path.exists(le_path):
        print("Error: label_encoder.pkl not found. Please train a model first.")
        print("  Run: python src/train.py --model cnn_lstm")
        sys.exit(1)

    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Load dataset
    if not os.path.exists(config.DATASET_PATH):
        print(f"Error: Dataset not found at {config.DATASET_PATH}")
        sys.exit(1)

    metadata_df = load_ravdess_metadata(config.DATASET_PATH)
    if len(metadata_df) == 0:
        print("Error: No audio files found in dataset.")
        sys.exit(1)

    _, _, test_df = actor_wise_split(
        metadata_df,
        train=config.TRAIN_SPLIT,
        val=config.VAL_SPLIT,
        test=config.TEST_SPLIT,
        seed=config.RANDOM_SEED
    )

    print(f"Extracting features for {len(test_df)} test samples...")
    X_test, y_test = prepare_dataset(test_df, config.DATASET_PATH, config, augment=False)
    y_test_encoded = label_encoder.transform(y_test)

    results = []
    all_eval = {}

    # Evaluate CNN
    cnn_path = os.path.join(config.MODELS_DIR, 'cnn_model.keras')
    print("\n--- Evaluating CNN model ---")
    cnn_results = evaluate_model(cnn_path, X_test, y_test_encoded, label_encoder)
    if cnn_results:
        print(f"CNN Accuracy: {cnn_results['accuracy']:.4f}")
        print(f"CNN F1-Score: {cnn_results['f1']:.4f}")
        print(cnn_results['report'])
        results.append({
            'Model': 'CNN',
            'Accuracy': cnn_results['accuracy'],
            'Precision': cnn_results['precision'],
            'Recall': cnn_results['recall'],
            'F1-Score': cnn_results['f1']
        })
        all_eval['cnn'] = cnn_results
    else:
        print("CNN model not found — skipping.")

    # Evaluate CNN-LSTM
    lstm_path = os.path.join(config.MODELS_DIR, 'cnn_lstm_model.keras')
    print("\n--- Evaluating CNN-LSTM model ---")
    lstm_results = evaluate_model(lstm_path, X_test, y_test_encoded, label_encoder)
    if lstm_results:
        print(f"CNN-LSTM Accuracy: {lstm_results['accuracy']:.4f}")
        print(f"CNN-LSTM F1-Score: {lstm_results['f1']:.4f}")
        print(lstm_results['report'])
        results.append({
            'Model': 'CNN-LSTM',
            'Accuracy': lstm_results['accuracy'],
            'Precision': lstm_results['precision'],
            'Recall': lstm_results['recall'],
            'F1-Score': lstm_results['f1']
        })
        all_eval['cnn_lstm'] = lstm_results
    else:
        print("CNN-LSTM model not found — skipping.")

    # Save comparison CSV
    if results:
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(config.RESULTS_DIR, 'model_comparison.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\nSaved comparison to {csv_path}")

    # Save full evaluation results as JSON
    if all_eval:
        eval_path = os.path.join(config.RESULTS_DIR, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(all_eval, f, indent=2)
        print(f"Saved full evaluation to {eval_path}")

    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
