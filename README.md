# 🎬 Movie Review Sentiment Analysis

An end-to-end deep learning project to analyze the sentiment of IMDb movie reviews using LSTM, Word2Vec embeddings, and a Flask web app.

## Overview

This project performs sentiment analysis on movie reviews from the IMDb dataset.
It classifies user reviews into Positive, Neutral, or Negative, assigns a star rating (0.5⭐ to 5⭐), and provides a probability score for prediction confidence.

The solution includes:
- Data download, preprocessing, and cleaning
- Word2Vec embeddings + LSTM-based deep learning
- Model training, evaluation, and deployment
- RESTful Flask web app for real-time predictions

![Movie Review Sentiment Analysis](https://github.com/Subith-Varghese/Moview_Review_Sentiment_Analysis/blob/f3e0d61795752d5c4a09b52507589565ad107025/Screenshot%202025-09-04%20202555.png)


---
## 📂 Project Structure

```
movie_review_sentiment_analysis/
│
├── app.py                         # Flask web app entry point
├── templates/
│   └── home.html                 # Frontend template for Flask app
│
├── src/
│   ├── components/               # Core ML components
│   │   ├── data_downloader.py    # Downloads IMDb dataset
│   │   ├── data_preprocessing.py # Cleans, lemmatizes, encodes labels
│   │   ├── data_splitter.py      # Splits train/test data
│   │   ├── embeddings_trainer.py # Tokenizer + Word2Vec embeddings
│   │   ├── model_trainer.py      # Trains LSTM model
│   │   ├── predictor.py          # Loads model & predicts sentiment
│   │   ├── evaluate_model.py     # Generates classification reports & confusion matrix
│   │
│   ├── pipeline/                 # Training & prediction pipelines
│   │   ├── train_pipeline.py     # Orchestrates end-to-end training
│   │   └── predict_pipeline.py   # CLI-based prediction pipeline
│   │
│   ├── utils/
│   │   └── logger.py            # Centralized logging utility
│
├── models/                      # Saved models, tokenizers, label encoders
├── data/
│   ├── imdb-dataset-of-50k-movie-reviews/
│   │   └── IMDB Dataset.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│
├── reports/
│   └── confusion_matrix.png     # Generated after evaluation
│
├── requirements.txt             # Required Python dependencies
└── README.md                   # Project documentation

```

## 🚀 Project Workflow
### 1. Data Download & Preprocessing

- Download IMDb dataset using opendatasets
- Handle missing values & duplicates
- Expand contractions, remove HTML, lowercase text
- Lemmatization + stopword removal (keeping negations like not, never)
- Encode sentiment labels (positive=1, negative=0)

### 2. Dataset Splitting
- Split into train (80%) & test (20%) sets

### 3. Embeddings Training

- Tokenize text & convert to sequences
- Train Word2Vec embeddings using Gensim
- Pad sequences for LSTM compatibility

### 4. LSTM Model Training

- Architecture:
  - Embedding Layer → LSTM Layer → Dense Output (Sigmoid)

- Loss: binary_crossentropy
- Optimizer: Adam
- Early stopping, model checkpoints, and LR scheduling used
- Best model saved as:

```
models/best_lstm_model.h5
```
### 5. Model Evaluation
- Evaluate on test set
- Generate classification report & confusion matrix
- Output stored in:

``` reports/confusion_matrix.png ```

### 6. Prediction

Two options available:
- ### CLI Pipeline
``` python -m src.pipeline.predict_pipeline ```

- ### Flask Web App

``` python app.py```

- ### Then open:

``` http://127.0.0.1:5000/```

## Installation & Setup

## Dataset Details

- Source: IMDb Movie Reviews Dataset
- Size: 50,000 movie reviews
- Columns:
  - review: Text of the review
  - sentiment: Positive / Negative
