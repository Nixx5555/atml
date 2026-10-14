"""
model.py - the deep autoencoder used by RikoNet (paper Sec. 3.1.1), built in TensorFlow/Keras.

Architecture (Kuchaiev & Ginsburg 2017, which RikoNet follows):

    input (n ratings [+ demographic features])
      -> Dense 512 (SELU) -> Dense 512 (SELU) -> Dense 1024 (SELU)   = encoder, code size d
      -> Dropout(0.8) on the code
      -> Dense 512 (SELU) -> Dense 512 (SELU)                        = decoder
      -> Dense n (ReLU)                                              = predicted ratings

Two training tricks:
  * Masked MSE (paper eq. 4): the error is only computed where a real rating exists, because a 0
    in the input means "unknown", not "rated zero".
  * Dense re-feeding (K&G): after a normal step, the dense output f(x) is fed back in as a new
    input whose target is f(x) itself. This teaches the network to be consistent on fully dense
    vectors, which is what it faces at prediction time.

The same class also builds the two embedding autoencoders of Sec. 3.1.2 (masked=False,
final_activation=None for the third one).
"""
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

import config as C


def build_autoencoder(n_in, n_out, hidden, activation, dropout, final_activation, l2=0.0):
    """Return (full autoencoder, encoder) Keras models sharing the same weights."""
    init = "lecun_normal" if activation == "selu" else "glorot_uniform"  # SELU needs lecun init
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    inp = tf.keras.Input(shape=(n_in,))
    x = inp
    for h in hidden:                                   # encoder
        x = tf.keras.layers.Dense(h, activation=activation, kernel_initializer=init,
                                  kernel_regularizer=reg)(x)
    code = x
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)        # regularise the code layer
    for h in reversed(hidden[:-1]):                    # decoder (mirror of encoder)
        x = tf.keras.layers.Dense(h, activation=activation, kernel_initializer=init,
                                  kernel_regularizer=reg)(x)
    out = tf.keras.layers.Dense(n_out, activation=final_activation, kernel_regularizer=reg)(x)
    return tf.keras.Model(inp, out), tf.keras.Model(inp, code)


class DeepAutoencoder:
    """Trainer wrapper: masked loss, dense re-feeding, early stopping, batched prediction."""

    def __init__(self, n_out, n_side=0, hidden=C.HIDDEN, activation=C.ACTIVATION,
                 dropout=C.DROPOUT, final_activation=C.FINAL_ACTIVATION, masked=True,
                 refeed=C.DENSE_REFEEDING, lr=C.LEARNING_RATE, l2=0.0, seed=C.SEED):
        tf.keras.utils.set_random_seed(seed)
        self.n_out, self.n_side, self.masked, self.refeed = n_out, n_side, masked, refeed
        self.model, self.encoder = build_autoencoder(n_out + n_side, n_out, hidden, activation,
                                                     dropout, final_activation, l2)
        self.opt = tf.keras.optimizers.Adam(lr)
        self.rng = np.random.default_rng(seed)
        self._step = tf.function(self._train_step)

    # ------------------------------------------------------------------ one optimisation step
    def _train_step(self, x, y):
        mask = tf.cast(y > 0, tf.float32) if self.masked else tf.ones_like(y)
        with tf.GradientTape() as tape:
            p = self.model(x, training=True)
            loss = tf.reduce_sum(mask * tf.square(y - p)) / tf.maximum(tf.reduce_sum(mask), 1.0)
            total = loss + tf.add_n(self.model.losses) if self.model.losses else loss
        grads = tape.gradient(total, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.refeed:  # dense re-feeding: input f(x), target f(x)
            target = tf.stop_gradient(p)
            x2 = tf.concat([target, x[:, self.n_out:]], axis=1) if self.n_side else target
            with tf.GradientTape() as tape:
                p2 = self.model(x2, training=True)
                loss2 = tf.reduce_mean(tf.square(target - p2))
            grads = tape.gradient(loss2, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # ------------------------------------------------------------------ helpers
    def _inputs(self, X, rows, side):
        x = X[rows].toarray() if sp.issparse(X) else X[rows]
        if self.n_side:
            x = np.hstack([x, side[rows]])
        return x.astype(np.float32)

    def predict(self, X, side=None, rows=None, batch_size=1024):
        """Predicted rating matrix for `rows` (default: all rows) given input matrix X."""
        rows = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
        out = [self.model(self._inputs(X, rows[s:s + batch_size], side), training=False).numpy()
               for s in range(0, len(rows), batch_size)]
        return np.vstack(out)

    def encode(self, X, side=None, batch_size=1024):
        rows = np.arange(X.shape[0])
        return np.vstack([self.encoder(self._inputs(X, rows[s:s + batch_size], side),
                                       training=False).numpy()
                          for s in range(0, len(rows), batch_size)])

    def masked_mse(self, X_in, X_target, side=None, batch_size=1024):
        """Masked MSE (paper eq. 4) on the non-zero entries of X_target, inputs from X_in."""
        X_target = sp.csr_matrix(X_target)
        rows = np.unique(X_target.nonzero()[0])
        se, n = 0.0, 0
        for s in range(0, len(rows), batch_size):
            r = rows[s:s + batch_size]
            p = self.predict(X_in, side, r, batch_size)
            t = X_target[r].toarray()
            m = t > 0
            se += float(((t - p)[m] ** 2).sum())
            n += int(m.sum())
        return se / n

    # ------------------------------------------------------------------ training loop
    def fit(self, X_train, X_val=None, side=None, epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE,
            patience=C.PATIENCE, verbose=True, label=""):
        """Train on X_train. If X_val is given, track its masked MSE (inputs = X_train) and stop
        early when it has not improved for `patience` epochs; the best weights are restored."""
        history = {"train_mse": [], "val_mse": []}
        best, best_w, wait = np.inf, None, 0
        n = X_train.shape[0]
        for ep in range(1, epochs + 1):
            order = self.rng.permutation(n)
            losses = [float(self._step(self._inputs(X_train, order[s:s + batch_size], side),
                                       self._target(X_train, order[s:s + batch_size])))
                      for s in range(0, n, batch_size)]
            history["train_mse"].append(float(np.mean(losses)))
            msg = f"[{label}] epoch {ep:3d}  train MSE {history['train_mse'][-1]:.4f}"
            if X_val is not None:
                v = self.masked_mse(X_train, X_val, side)
                history["val_mse"].append(v)
                msg += f"  val MSE {v:.4f}"
                if v < best - 1e-5:
                    best, best_w, wait = v, self.model.get_weights(), 0
                else:
                    wait += 1
            if verbose and (ep == 1 or ep % 5 == 0):
                print(msg, flush=True)
            if X_val is not None and wait >= patience:
                if verbose:
                    print(f"[{label}] early stop at epoch {ep}; best val MSE {best:.4f}", flush=True)
                break
        if best_w is not None:
            self.model.set_weights(best_w)
        history["best_epoch"] = int(np.argmin(history["val_mse"]) + 1) if history["val_mse"] else ep
        return history

    def _target(self, X, rows):
        y = X[rows].toarray() if sp.issparse(X) else X[rows]
        return y.astype(np.float32)
