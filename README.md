# Tweets-Mood-Analysis

## Goal

This topic is subject to various contests, and in particular SemEval2019 - Task #3, which stemmed this very project. The goal is to classify tweet into 4 categories: **angry**, **sad**, **others**, and **happy**.

The task has been completed using NLP methods and convolutional neural networks.

The main issue is that there are very few tweets labeled with these 4 mood above, but on the other hand there are a lot of tweets labeled for **positive** and **negative** tweets. There is also a lot of tweets labeled with these 6 categories: **angry**, **sad**, **fear**, **surprise**, **joy**, and **love**.
So, a part of this project is to apply a transfer learning with a neural network trained with tweets labeled at 2 polarities onto another neural network for tweets labeled at 6 mood categories in a first part. And then, apply a transfer learning onto another neural network for tweets labeled at 4 mood categories.

## Resume

This program has been developed with Python programming language, and use Keras with TensorFlow backend. Some part of this code is provided from [this Github project](https://github.com/abdulfatir/twitter-sentiment-analysis).

## Datasets

- `dataset/tweets/train_moods.txt` contained **30,159** tweets labeled with **angry**, **sad**, **other** and **happy**.
- `dataset/tweets/train_0_to_4.csv` contained **1,600,000** tweets labeled with polarity from **0 (negative)** to **4 (positive)**.
    
    This dataset is from kaggle: https://www.kaggle.com/kazanova/sentiment140
- `dataset/tweets/train_positive_negative.csv` contained **100,000** tweets labeled with **0 (negative)** and **1 (positive)** polarities.

    This dataset is from kaggle: https://www.kaggle.com/imrandude/twitter-sentiment-analysis


## Installation

- Go to `dataset/embedding/README.md` and follow the instruction to download the embedding matrix.

- Go to `dataset/tweets/README.md` and follow the instruction to download the embedding matrix.

## Execution

TODO

## Bibliography

TODO


## Authors

EPITA School, SCIA Master 2 - Project for Deep Learning and Natural Language Processing Course. 

Authors: 
- **BENSABAT David** (bensab_d)
- **YVONNE Xavier** (xavier.yvonne)