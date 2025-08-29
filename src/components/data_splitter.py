import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger

class DataSplitter:
    def __init__(self, df: pd.DataFrame, test_ratio=0.2, random_state=42):
        self.df = df
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split_and_save(self, output_dir="data/processed"):
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Split train & temp first
            train_df, test_df = train_test_split(
                self.df,
                test_size=self.test_ratio,
                random_state=self.random_state
            )
            # Save splits
            train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

            logger.info(f"✅ Dataset successfully split & saved in {output_dir}")
            logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            logger.exception(f"❌ Error while splitting dataset: {e}")
            raise e