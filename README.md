# Named Entity Recognition (NER) with Bi-LSTM

## Overview

This project implements Named Entity Recognition (NER) using a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network. The goal is to identify and classify entities such as persons, organizations, and locations in a given text dataset!

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)

## Introduction

Named Entity Recognition is a natural language processing task that involves identifying and classifying named entities in text. In this project, we leverage a Bi-LSTM model, a type of recurrent neural network, to perform NER on a given dataset.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/shlok191/NLP-Named-Entity-Recognition-Model
   cd NLP-Named-Entity-Recognition-Model
   pip install -r requirements.txt
   ```
   
2. Utilize python3 to run the main.py file

## Dataset

The Named Entity Recognition (NER) model in this project is trained and evaluated on the CoNLL-2003 dataset. The CoNLL-2003 dataset is a widely used benchmark for NER tasks and consists of news articles from the Reuters corpus. It provides labeled entities such as persons, organizations, and locations. (Sample: https://huggingface.co/datasets/conll2003)


## Training

### Prepare Your Dataset

Before training the Bi-LSTM model, ensure your dataset is prepared in the required format. For the CoNLL-2003 dataset, follow the preprocessing steps mentioned in the [Dataset](#dataset) section.

### Execute Training Script

To train the Named Entity Recognition (NER) model, use the provided training script. The script takes care of loading the preprocessed dataset, configuring the Bi-LSTM model, and training it.

```bash
python train.py --data_path ./CoNLL-2003/training
python train.py --data_path ./CoNLL-2003/training --embedding_dim 100 --hidden_dim 64
```

## Results

### Performance Metrics

After training and evaluating your Bi-LSTM NER model, you can analyze its performance using various metrics, including precision, recall, and F1 score. These metrics help assess how well the model identifies and classifies named entities.

To evaluate the model, follow these steps:

1. Ensure you've already trained the model using the instructions provided in the [Training](#training) section.

2. Evaluate the model's performance on a test dataset by running the following command:

   ```bash
   python evaluate.py --model_path path/to/your/trained/model --test_data_path path/to/test/dataset

## Results

### Performance Metrics

After training and evaluating your Bi-LSTM NER model, you can analyze its performance using various metrics, including precision, recall, and F1 score. These metrics help assess how well the model identifies and classifies named entities.

To evaluate the model, follow these steps:

1. Ensure you've already trained the model using the instructions provided in the [Training](#training) section.

2. Evaluate the model's performance on a test dataset by running the following command:

   ```bash
   python evaluate.py --model_path path/to/your/trained/model --test_data_path path/to/test/dataset
   ```

Metrics on Test Set:
- Precision: 0.92
- Recall: 0.88
- F1 Score: 0.90
