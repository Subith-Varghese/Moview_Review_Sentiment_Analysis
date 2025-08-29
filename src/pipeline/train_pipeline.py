from src.utils.logger import logger
from src.components.data_downloader import DataDownloader
from src.components.data_preprocessing import DataPreprocessing
from src.components.embeddings_trainer import EmbeddingsTrainer
from src.components.model_trainer import ModelTrainer
from src.components.data_splitter import DataSplitter
import pandas as pd 
import os


if __name__ == "__main__":
    try:   
        # Step 1: Download dataset
        dataset_url = "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
        downloader = DataDownloader(dataset_url, "data/")
        logger.info(f"📥 Downloading dataset from {dataset_url} ...")
        downloader.download()
        logger.info(f"✅ Dataset downloaded successfully.")

        # Step 2: Dataset base directory
        csv_path = os.path.join("data", "imdb-dataset-of-50k-movie-reviews", "IMDB Dataset.csv")
        df = pd.read_csv(csv_path)
        logger.info(f"📄 Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Step 3: Preprocess dataset
        preprocessing = DataPreprocessing(df)

        # Step 3: Download NLTK data (only in training, not in prediction)
        preprocessing.download_nltk_data()
        # Handle missing values & duplicates
        df = preprocessing.handle_missing_and_duplicates()
        # Encode sentiment labels 
        df = preprocessing.encode_labels(target_column="sentiment")
        # Clean review text
        df = preprocessing.apply_cleaning(text_column="review")
        logger.info("✅ Data preprocessing completed.")

        # Step 4: Split dataset into train, val, test
        splitter = DataSplitter(df)
        train_df,test_df = splitter.split_and_save("data/processed")

        # STEP 5: TRAIN EMBEDDINGS (TOKENIZER + WORD2VEC + PADDING)
        logger.info("🔠 Training tokenizer & Word2Vec embeddings...")
        embeddings_trainer = EmbeddingsTrainer(train_df, test_df)
        X_train_pad, X_test_pad, y_train, y_test, tokenizer, embedding_matrix = embeddings_trainer.train_embeddings()
        logger.info("✅ Embedding training completed.")

        # STEP 6: TRAIN LSTM MODEL
        logger.info("🤖 Training LSTM model...")
        model_trainer = ModelTrainer(
            embedding_matrix=embedding_matrix,
            X_train=X_train_pad,
            y_train=y_train,
            X_test=X_test_pad,
            y_test=y_test,
            epoch = 15
        )

        model, history = model_trainer.train_model()
        logger.info("✅ LSTM model training completed successfully.")

    except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            print(f"[ERROR] Training pipeline failed: {e}")
