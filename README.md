# Tweets-Mood-Analysis

## Goal

This topic is subject to various contests, and in particular SemEval2019 - Task #3, which stemmed this very project. The goal is to classify tweet into 4 categories: **angry**, **sad**, **others**, and **happy**.

The task has been completed using NLP methods and convolutional neural networks.

The main issue is that there are very few tweets labeled with these 4 mood above, but on the other hand there are a lot of tweets labeled for **positive** and **negative** tweets. There is also a lot of tweets labeled with these 6 categories: **angry**, **sad**, **fear**, **surprise**, **joy**, and **love**.
So, a part of this project is to apply a transfer learning with a neural network trained with tweets labeled at 2 polarities onto another neural network for tweets labeled at 6 mood categories in a first part. And then, apply a transfer learning onto another neural network for tweets labeled at 4 mood categories.

## Resume

This program has been developed with Python programming language, and use Keras with TensorFlow backend. Some part of this code is provided from [this Github project](https://github.com/abdulfatir/twitter-sentiment-analysis).

## Datasets

- `./dataset/tweets/tweets_polarity_2/tweets_pos_neg_train-processed.csv` contained **640,000** tweets labeled with polarities **0 (negative)** and **1 (positive)**.

- `./dataset/tweets/tweets_emotion_6/emotion_6-processed.csv` contained **416,809** tweets labeled with **0 (angry)**, **1 (sad)**, **2 (fear)**, **3 (surprise)**, **4 (joy)** and **5 (love)**.

- `./dataset/tweets/tweets_emotion_4/tweets_emotions_train-processed.csv` contained **30,159** tweets labeled with **angry**, **sad**, **other** and **happy**.

## Requirements

There are some general library requirements for the project:
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`
* `keras` with `TensorFlow`

**Note**: It is recommended to use Anaconda distribution of Python.

## Installation

- Go to `dataset/embedding/README.md` and follow the instruction to download the embedding matrix.

## Execution

- Post preprocess is done by `./src/post_preprocess.ipynb` (already done)
- Preprocess is done by `./src/preprocess.py` (already done)
- First model (polarities negative/positive) is done by `./src/cnn.py`
- Second and third model (emotions 6 and 4) is done by `./src/transfer_learning.ipynb`

Each running of CNN Will validate using 10% data and save models for each epoch in `./models/`. (Please make sure this directory exists before running `cnn.py`). 

- Write predictions using the best model at the end of `./src/transfer_learning.ipynb`
- Our best model is save at `./models/transfer_emo6_emo4/entire_corpus/20-0.921-0.608-0.937-0.592.hdf5`

## Authors

EPITA School, SCIA Master 2 - Project for Deep Learning and Natural Language Processing Course. 

Authors: 
- **BENSABAT David** (bensab_d)
- **YVONNE Xavier** (xavier.yvonne)