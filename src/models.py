"""
EMOTIVA — Deep Learning Model Architectures for Speech Emotion Recognition.
Provides CNN baseline and CNN-LSTM architectures.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(128, 130, 1), num_classes=8):
    """
    CNN baseline architecture.
    Conv2D -> ReLU -> MaxPool -> Dropout -> Conv2D -> ReLU -> MaxPool -> Dropout -> Conv2D -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax.
    """
    inputs = layers.Input(shape=input_shape, name='mel_spectrogram_input')

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = layers.Dropout(0.2, name='drop3')(x)

    # Flatten & Dense Layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='drop_dense')(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax', name='cnn_softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='cnn_model')
    return model


def build_cnn_lstm_model(input_shape=(128, 130, 1), num_classes=8):
    """
    CNN-LSTM hybrid architecture.
    CNN feature extractor -> Permute(time, freq, ch) -> Reshape -> LSTM -> Dense -> Softmax.
    Preserves chronological time frames along the temporal sequence axis.
    """
    inputs = layers.Input(shape=input_shape, name='mel_spectrogram_input')

    # CNN Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)

    # CNN Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = layers.Dropout(0.2, name='drop2')(x)

    # CNN Block 3
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = layers.Dropout(0.2, name='drop3')(x)

    # Permute to (batch, time, freq, channels) so time is the sequence dimension
    # Shape after 3 MaxPool(2,2): (batch, freq=16, time=16, channels=64)
    x = layers.Permute((2, 1, 3), name='permute_time_first')(x)
    time_steps = x.shape[1]
    feature_dim = x.shape[2] * x.shape[3]
    x = layers.Reshape((time_steps, feature_dim), name='temporal_reshape')(x)

    # Temporal Sequence Modeling
    x = layers.LSTM(64, return_sequences=False, name='lstm_temporal')(x)
    x = layers.Dropout(0.3, name='drop_lstm')(x)

    # Dense Projection & Classification
    x = layers.Dense(64, activation='relu', name='dense_projection')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='emotion_softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='cnn_lstm_model')
    return model


def get_model_summary(model: models.Model) -> str:
    """Return string summary of the model."""
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return '\n'.join(summary)
