import re
import os
import pickle
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import logger


class DataPreprocessing:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoder = LabelEncoder()

        # Stopwords but keep negations important for sentiment
        stop_words = set(stopwords.words("english"))
        negations_to_keep = {
            "not", "no", "nor", "never", "none", "nobody", "nothing", "neither",
            "cannot", "without", "hardly", "scarcely", "barely",
            "can", "do", "does", "did", "is", "are", "was", "were",
            "would", "should", "could", "had", "has", "have", "ain"
        }
        self.stop_words = stop_words - negations_to_keep

    def handle_missing_and_duplicates(self):
        logger.info("🔍 Checking missing values and duplicates...")

        # Drop missing values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            logger.warning(f"⚠️ Missing values detected: \n{null_counts}")
            self.df.dropna(inplace=True)
            logger.info("✅ Missing values removed.")

        # Drop duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"⚠️ Found {duplicates} duplicate rows. Removing...")
            self.df.drop_duplicates(inplace=True)
            logger.info("✅ Duplicates removed.")

        logger.info(f"Final dataset shape after cleaning: {self.df.shape}")
        return self.df
    
    def encode_labels(self, target_column: str):
        """Encode sentiment labels into numerical format."""
        logger.info(f"🔄 Encoding target column '{target_column}'...")
        self.df[target_column] = self.label_encoder.fit_transform(self.df[target_column])
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        # Save label encoder
        with open("models/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"✅ Labels encoded successfully: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        logger.info("💾 LabelEncoder saved to 'models/label_encoder.pkl'")
        return self.df

    def clean_text(self, text: str):
        # Expand contractions
        text = contractions.fix(str(text))
        # Lowercase
        text = text.lower()
        # Remove HTML tags & URLs
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        # Tokenize words
        tokens = word_tokenize(text)
        # Remove punctuations
        tokens = [word for word in tokens if word not in string.punctuation]
        # Lemmatize & remove stopwords
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens if word not in self.stop_words
        ]
        return " ".join(tokens)

    def apply_cleaning(self, text_column: str):
        logger.info(f"🧹 Cleaning text column '{text_column}'...")
        # Create a new column instead of overwriting the original
        clean_col = f"clean_{text_column}"
        self.df[clean_col] = self.df[text_column].apply(self.clean_text)
        logger.info(f"✅ Text cleaning completed. Cleaned text stored in '{clean_col}'.")
        return self.df
    
    @staticmethod
    def download_nltk_data():
        nltk.download("stopwords")
        nltk.download("punkt_tab")
        nltk.download("wordnet")
        logger.info("✅ NLTK resources are downloaded.")
