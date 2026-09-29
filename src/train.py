"""
EMOTIVA — Training script.
Trains CNN or CNN-LSTM models on the RAVDESS dataset.

Usage:
    python src/train.py --model cnn
    python src/train.py --model cnn_lstm
    python src/train.py --model cnn_lstm --augment
    python src/train.py --model cnn_lstm --augment --epochs 100
"""
import os
import sys
import argparse
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import load_ravdess_metadata, actor_wise_split
from src.feature_extraction import prepare_dataset
from src.models import build_cnn_model, build_cnn_lstm_model


def main():
    parser = argparse.ArgumentParser(description="Train EMOTIVA SER Model")
    parser.add_argument('--model', type=str, choices=['cnn', 'cnn_lstm'],
                        default='cnn_lstm', help="Model architecture to train")
    parser.add_argument('--augment', action='store_true',
                        help="Enable data augmentation for training")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Training batch size (default 16 for optimal gradient updates)")
    args = parser.parse_args()

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  EMOTIVA — Speech Emotion Recognition Training")
    print(f"  Model: {args.model.upper()}")
    print(f"  Augmentation: {'ON' if args.augment else 'OFF'}")
    print(f"  Max Epochs: {args.epochs}")
    print("=" * 60)

    # 1. Load dataset metadata
    print("\n[1/6] Loading dataset metadata...")
    if not os.path.exists(config.DATASET_PATH):
        print(f"\nError: Dataset not found at {config.DATASET_PATH}")
        print("Please download the RAVDESS dataset and place audio files in:")
        print(f"  {config.DATASET_PATH}/Actor_01/*.wav")
        print(f"  {config.DATASET_PATH}/Actor_02/*.wav")
        print("  ...")
        print(f"  {config.DATASET_PATH}/Actor_24/*.wav")
        print("\nDownload from: https://zenodo.org/record/1188976")
        sys.exit(1)

    metadata_df = load_ravdess_metadata(config.DATASET_PATH)
    if len(metadata_df) == 0:
        print("Error: No valid .wav files found in dataset path.")
        sys.exit(1)

    print(f"  Found {len(metadata_df)} audio files.")
    print(f"  Actors: {metadata_df['actor'].nunique()}")
    print(f"  Emotion distribution:")
    for emo_id, count in metadata_df['emotion'].value_counts().sort_index().items():
        emo_name = config.EMOTION_LABELS.get(emo_id, str(emo_id))
        print(f"    {emo_name}: {count}")

    # 2. Split dataset (actor-wise)
    print("\n[2/6] Splitting dataset (actor-wise to prevent speaker leakage)...")
    train_df, val_df, test_df = actor_wise_split(
        metadata_df,
        train=config.TRAIN_SPLIT,
        val=config.VAL_SPLIT,
        test=config.TEST_SPLIT,
        seed=config.RANDOM_SEED
    )

    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(metadata_df)*100:.1f}%)")

    # 3. Extract features
    print("\n[3/6] Extracting features for training set...")
    X_train, y_train = prepare_dataset(train_df, config.DATASET_PATH, config, augment=args.augment)
    print(f"  Training samples: {len(X_train)} (includes augmented)" if args.augment else f"  Training samples: {len(X_train)}")

    print("\n[4/6] Extracting features for validation set...")
    X_val, y_val = prepare_dataset(val_df, config.DATASET_PATH, config, augment=False)
    print(f"  Validation samples: {len(X_val)}")

    # 4. Encode labels (fitted on all 8 classes for strict consistency)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(config.EMOTION_LABELS.keys()))
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Save label encoder
    encoder_path = os.path.join(config.MODELS_DIR, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"\n  Label encoder saved to {encoder_path}")
    print(f"  Classes: {list(label_encoder.classes_)}")

    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
    y_val_cat = to_categorical(y_val_encoded, num_classes=num_classes)

    # 5. Build model
    input_shape = X_train.shape[1:]
    print(f"\n[5/6] Building {args.model.upper()} model...")
    print(f"  Input shape: {input_shape}")

    if args.model == 'cnn':
        model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    else:
        model = build_cnn_lstm_model(input_shape=input_shape, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 6. Train
    model_save_path = os.path.join(config.MODELS_DIR, f"{args.model}_model.keras")
    callbacks = [
        EarlyStopping(monitor='loss', patience=12, min_delta=0.005, verbose=1),
        ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor='accuracy', save_best_only=True, verbose=1)
    ]

    print(f"\n[6/6] Training {args.model.upper()} model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure model is saved if ModelCheckpoint did not trigger
    if not os.path.exists(model_save_path):
        model.save(model_save_path)

    # Save training history as JSON
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    history_path = os.path.join(config.RESULTS_DIR, f"{args.model}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Model saved to: {model_save_path}")
    print(f"  History saved to: {history_path}")
    print(f"  Label encoder saved to: {encoder_path}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  Evaluate: python src/evaluate.py")
    print(f"  Run app:  streamlit run app.py")


if __name__ == '__main__':
    main()
