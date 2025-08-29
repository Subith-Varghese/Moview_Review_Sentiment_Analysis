import numpy as np
import os
import pickle
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.logger import logger

class EmbeddingsTrainer:
    def __init__(self, train_df, test_df, max_vocab=20000, max_seq_len=200):
        self.train_df = train_df
        self.test_df = test_df
        self.MAX_VOCAB_SIZE = max_vocab
        self.MAX_SEQUENCE_LENGTH = max_seq_len

    def train_embeddings(self):
        logger.info("🧠 Training tokenizer...")
        tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(self.train_df["clean_review"])

        # Convert to sequences
        X_train_seq = tokenizer.texts_to_sequences(self.train_df["clean_review"])
        X_test_seq = tokenizer.texts_to_sequences(self.test_df["clean_review"])

        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

        y_train = self.train_df["sentiment"].values
        y_test = self.test_df["sentiment"].values

        # Save the tokenizer
        os.makedirs("models", exist_ok=True)
        tokenizer_path = os.path.join("models", "tokenizer.pkl")
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        logger.info(f"💾 Tokenizer saved at: {tokenizer_path}")

        # Train Word2Vec embeddings on top words
        logger.info("⚡ Training Word2Vec model...")
        top_words = set(list(tokenizer.word_index.keys())[:self.MAX_VOCAB_SIZE])
        filtered_sentences = self.train_df["clean_review"].apply(
            lambda doc: [word for word in doc.split() if word in top_words]
        )

        model_vec = gensim.models.Word2Vec(
            sentences=filtered_sentences,
            vector_size=300,
            window=5,
            min_count=1,
            workers=4
        )

        # Build embedding matrix
        logger.info("🔄 Building embedding matrix...")
        embedding_dim = 300
        word_index = tokenizer.word_index
        num_words = self.MAX_VOCAB_SIZE + 1

        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, i in word_index.items():
            if i < num_words and word in model_vec.wv:
                embedding_matrix[i] = model_vec.wv[word]

        logger.info("✅ Embedding matrix built successfully.")
        return X_train_pad, X_test_pad, y_train, y_test, tokenizer, embedding_matrix
