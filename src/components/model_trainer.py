from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.utils.logger import logger
import os 
import pickle

class ModelTrainer:
    def __init__(self, embedding_matrix, X_train, y_train, X_test, y_test,epoch=8):
        self.embedding_matrix = embedding_matrix
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.MAX_SEQUENCE_LENGTH = X_train.shape[1]
        self.epoch = epoch

    def build_lstm_model(self):
        model = Sequential()
        model.add(Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            input_length=self.MAX_SEQUENCE_LENGTH,
            trainable=False
        ))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train_model(self):
        logger.info("🚀 Starting LSTM training...")
        model = self.build_lstm_model()

        checkpoint_path = "models/best_lstm_model.h5"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
        ]

        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epoch,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # Save the history
        os.makedirs("models", exist_ok=True)
        history_path = os.path.join("models", "training_history.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(history.history, f)
        logger.info("💾 Training history saved to 'models/training_history.pkl'")
        logger.info("✅ Model training completed successfully!")
        return model, history
