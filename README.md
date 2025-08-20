# Movie Review Sentiment Analysis (LSTM + Word2Vec)

A production-ready, MLOps-style project structured like your `face_mask_detector` repo,
adapted for NLP sentiment analysis on the IMDB dataset.

## Structure
```
movie_review_sentiment_mlopstyle/
│── data/                      # Place IMDB csv here (IMDB Dataset.csv)
│── models/                    # Saved models, tokenizer, embeddings, etc.
│── notebooks/                 # Jupyter notebooks (EDA / experiments)
│── src/
│   ├── components/            # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── embeddings_trainer.py
│   │   ├── model_builder.py
│   │   ├── model_trainer.py
│   │   └── predictor.py
│   │
│   ├── pipeline/              # Training & prediction pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   └── utils/
│       ├── common.py
│       └── logger.py
│
│── templates/                 # Flask HTML templates
│   └── home.html
│
│── app.py                     # Flask web app entry
│── config.yaml                # Paths & hyperparameters
│── requirements.txt           # Dependencies
│── README.md
```

## Quickstart
1) Put `IMDB Dataset.csv` under `data/`.
2) Create/activate a Python 3.10+ env and install deps:
   ```bash
   pip install -r requirements.txt
   ```
3) Train:
   ```bash
   python -m src.pipeline.train_pipeline
   ```
4) Predict (CLI demo):
   ```bash
   python -m src.pipeline.predict_pipeline --text "This movie was surprisingly great!"
   ```
5) Run web app:
   ```bash
   python app.py
   ```
