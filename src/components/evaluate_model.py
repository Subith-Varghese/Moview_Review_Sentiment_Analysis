import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from src.utils.logger import logger
import os

MODEL_PATH = "models/best_lstm_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
TEST_DATA_PATH = "data/processed/test.csv"

def evaluate_model():
    try:
        logger.info("🔍 Starting model evaluation...")

        # Load model
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully.")

        # Load tokenizer
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        # Load label encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        # Load test dataset
        test_df = pd.read_csv(TEST_DATA_PATH)
        X_test = test_df["clean_review"].values
        y_test = test_df["sentiment"].values

        # Convert text to padded sequences
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding="post", truncating="post")

        # Evaluate model
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
        logger.info(f"✅ Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

        # Predictions
        y_pred_probs = model.predict(X_test_pad)
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        logger.info("\n" + report)

        
        # Create folder to save confusion matrix
        os.makedirs("reports", exist_ok=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        # Save as PNG
        cm_path = os.path.join("reports", "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        
        plt.show()
        plt.close()

        logger.info(f"📊 Confusion matrix saved at: {cm_path}")
        logger.info("🎯 Model evaluation completed successfully.")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        print(f"[ERROR] Evaluation failed: {e}")

if __name__ == "__main__":
    evaluate_model()
