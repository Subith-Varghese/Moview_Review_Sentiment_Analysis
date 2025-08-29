import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from src.utils.logger import logger
from src.components.data_preprocessing import DataPreprocessing


class Predictor:
    def __init__(self):
        try:
            # Load the trained model
            model_path = os.path.join("models", "best_lstm_model.h5")
            self.model = load_model(model_path)
            logger.info(f"✅ Model loaded successfully from {model_path}")

            # Load tokenizer
            tokenizer_path = os.path.join("models", "tokenizer.pkl")
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")

            # Load label encoder
            label_encoder_path = os.path.join("models", "label_encoder.pkl")
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"✅ Label encoder loaded from {label_encoder_path}")

            # Initialize Data Preprocessor (for cleaning text before prediction)
            self.preprocessor = DataPreprocessing()

            # Set max sequence length (must match training)
            self.max_sequence_length = 200

        except Exception as e:
            logger.error(f"❌ Failed to initialize Predictor: {e}")
            raise e

    def predict(self, review):
        """
        Predict sentiment for single or multiple texts.
        :param texts: str or list of str
        :return: list of dicts with text, predicted label, and probability
        """
        try:
            cleaned_review = self.preprocessor.clean_text(review)

            # Convert text to sequences
            sequences = self.tokenizer.texts_to_sequences([cleaned_review])
            padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding="post", truncating="post")

            # Predict probabilities
            pred_prob = self.model.predict(padded)[0][0]

            rating_dict = {
                0.5: "⯪☆☆☆☆",
                1.0: "★☆☆☆☆",
                1.5: "★⯪☆☆☆",
                2.0: "★★☆☆☆",
                2.5: "★★⯪☆☆",
                3.0: "★★★☆☆",
                3.5: "★★★⯪☆",
                4.0: "★★★★☆",
                4.5: "★★★★⯪",
                5.0: "★★★★★"
            }
            

                    # Map probability → rating
            if pred_prob < 0.1:
                rating = 0.5
            elif pred_prob < 0.2:
                rating = 1.0
            elif pred_prob < 0.3:
                rating = 1.5
            elif pred_prob < 0.4:
                rating = 2.0
            elif pred_prob < 0.5:
                rating = 2.5
            elif pred_prob < 0.6:
                rating = 3.0
            elif pred_prob < 0.7:
                rating = 3.5
            elif pred_prob < 0.8:
                rating = 4.0
            elif pred_prob < 0.9:
                rating = 4.5
            else:
                rating = 5.0

            return rating_dict[rating],pred_prob,rating

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise e
