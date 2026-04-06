"""Stacked-LSTM primary model (TensorFlow / Keras).

Architecture (unchanged from the coursework): LSTM(128, return_sequences) -> BatchNorm ->
Dropout -> LSTM(64) -> Dropout -> Dense(32, relu) -> Dense(1, sigmoid). Binary cross-entropy,
Adam, early stopping on a chronological validation tail, LR halving on plateau.
"""

from __future__ import annotations

import os

import numpy as np

from sysml.config import LSTMConfig


def _import_tf():
    # Must be set before TensorFlow is imported: forces deterministic GPU kernels so that a
    # fixed seed gives reproducible runs (CUDA parallelism otherwise perturbs results).
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for primary.type='lstm'. "
            "Install it with `pip install '.[tf]'`, or use primary.type=logreg."
        ) from exc
    return tf


class LSTMPrimary:
    def __init__(self, cfg: LSTMConfig, seq_len: int, n_features: int, seed: int = 42):
        self.cfg, self.seq_len, self.n_features, self.seed = cfg, seq_len, n_features, seed
        self.history: dict[str, list[float]] | None = None
        self.model = None

    def _build(self):
        tf = _import_tf()
        tf.keras.utils.set_random_seed(self.seed)  # python, numpy and TF seeds at once
        layers, c = tf.keras.layers, self.cfg
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(self.seq_len, self.n_features)),
                # return_sequences=True hands the full sequence to the second LSTM.
                layers.LSTM(c.units1, return_sequences=True),
                layers.BatchNormalization(),
                layers.Dropout(c.dropout),
                layers.LSTM(c.units2),
                layers.Dropout(c.dropout),
                layers.Dense(c.dense, activation="relu"),
                layers.Dense(1, activation="sigmoid"),  # P(trend up)
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(c.lr), loss="binary_crossentropy")
        return model

    def fit(self, x, y, class_weight=None, epochs=None):
        tf = _import_tf()
        c = self.cfg
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=c.patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=c.lr_factor, patience=c.lr_patience, min_lr=c.min_lr
            ),
        ]
        self.model = self._build()
        # validation_split takes the *last* fraction of the data (no shuffling), so the early
        # stopping signal comes from the most recent part of the training block only.
        hist = self.model.fit(
            np.asarray(x, dtype="float32"),
            np.asarray(y, dtype="float32"),
            validation_split=c.val_split,
            epochs=epochs or c.epochs_final,
            batch_size=c.batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0,
            shuffle=True,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def predict_proba(self, x):
        assert self.model is not None, "call fit() first"
        return self.model.predict(np.asarray(x, dtype="float32"), verbose=0).ravel()
