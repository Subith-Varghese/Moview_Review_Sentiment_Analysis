# Movie Review Sentiment Analysis — Class-Based MLOps

A production-ready, **class-based** project mirroring your `face_mask_detector` structure, adapted for NLP (IMDB reviews) with **LSTM + Word2Vec**.

## Structure
```
movie_review_sentiment_mlopstyle_cb/
│── data/                      # Put IMDB Dataset.csv or use downloader
│── models/                    # Saved models, tokenizer, embeddings, logs
│── notebooks/                 # Optional notebooks
│── src/
│   ├── components/
│   │   ├── data_downloader.py       # Download from Kaggle/URL (opendatasets)
│   │   ├── data_ingestion.py        # Class: load + optional split
│   │   ├── data_preprocessing.py    # Class: clean + label encode
│   │   ├── embeddings_trainer.py    # Class: train/save Word2Vec
│   │   ├── model_builder.py         # Class: build LSTM from embedding matrix
│   │   ├── model_trainer.py         # Class: tokenizer, sequences, callbacks, train
│   │   └── predictor.py             # Class: load artifacts, predict text
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py        # Orchestrates full pipeline (classes)
│   │   └── predict_pipeline.py      # CLI to predict one review
│   │
│   └── utils/
│       ├── common.py                # read_config, save/load joblib, ensure_dir
│       └── logger.py                # module-level logger + get_logger()
│
│── templates/
│   └── home.html                    # Simple Flask UI
│
│── app.py                           # Flask app entry
│── config.yaml                      # Paths + hyperparameters
│── requirements.txt                 # Dependencies
│── README.md
```
## Quickstart
1) Option A — **Auto download** (needs Kaggle API configured): set `data.dataset_url` in `config.yaml`, then run training (downloader runs first).  
2) Option B — **Manual**: put `IMDB Dataset.csv` under `data/`.

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python -m src.pipeline.train_pipeline
```

### Predict (CLI)
```bash
python -m src.pipeline.predict_pipeline --text "This was a surprisingly great movie!"
```

### Run Web App
```bash
python app.py
```
