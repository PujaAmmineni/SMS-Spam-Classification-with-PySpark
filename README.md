# SMS-Spam-Classification-with-PySpark


This project implements various machine learning models for the classification of SMS messages into spam or non-spam categories. The models are evaluated based on their performance using various metrics, such as accuracy, sensitivity, specificity, and the area under the ROC curve (AUC). The goal is to identify the best-performing model for SMS spam detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Best Model](#best-model)


## Project Overview

This project involves building and evaluating the following machine learning models for SMS spam classification:
1. Naive Bayes Classifier
2. Random Forest Classifier
3. Support Vector Classifier (SVC)
4. K-Nearest Neighbors (KNN)
5. Logistic Regression

Each model is trained on a preprocessed dataset of SMS messages and evaluated using metrics such as accuracy, sensitivity, specificity, and AUC to determine the best model for classification.

## Setup and Installation

To get started with this project, ensure that you have the following installed:

- [Apache Spark](https://spark.apache.org/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/)
- [Python 3](https://www.python.org/) 
- [NLTK](https://www.nltk.org/)


### 3. Start Spark Session
To use Spark, you’ll need to start a Spark session in your Python environment. You can do this with the following code:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SMS Spam Classification").getOrCreate()
```

## Data Preprocessing

The preprocessing steps include:

1. **Data Loading**: The dataset is loaded from a CSV file into a PySpark DataFrame.
2. **Renaming Columns**: The dataset columns are renamed to `input` (SMS text) and `output` (spam label).
3. **Dropping Irrelevant Columns**: Unnecessary columns are removed to clean up the dataset.
4. **Handling Null/Empty Values**: Rows with missing or empty SMS text are filtered out.
5. **Text Cleaning**: The text data is cleaned by removing non-alphabetical characters, converting text to lowercase, and stemming words.
6. **Tokenization**: The text is split into tokens (words).
7. **Count Vectorization**: The tokenized words are converted into feature vectors using `CountVectorizer`.

## Model Training and Evaluation

Five machine learning models were trained and evaluated:

### 1. **Naive Bayes Classifier**
   - **Type**: Multinomial Naive Bayes
   - **Accuracy**: 98%
   - **Sensitivity**: 90.5%
   - **Specificity**: 99.2%
   - **AUC**: 0.9836

### 2. **Random Forest Classifier**
   - **Number of Trees**: 100
   - **Max Depth**: 10
   - **Accuracy**: 93.6%
   - **Sensitivity**: 53.1%
   - **Specificity**: 100%
   - **AUC**: 0.9804

### 3. **Support Vector Classifier (SVC)**
   - **Max Iterations**: 100
   - **Accuracy**: 97.6%
   - **Sensitivity**: 83.7%
   - **Specificity**: 99.78%
   - **AUC**: 0.9894

### 4. **K-Nearest Neighbors (KNN)**
   - **Neighbors**: 5
   - **Accuracy**: 97.3%
   - **Sensitivity**: 80.3%
   - **Specificity**: 100%
   - **AUC**: 0.9876

### 5. **Logistic Regression**
   - **Max Iterations**: 100
   - **Accuracy**: 97.3%
   - **Sensitivity**: 80.3%
   - **Specificity**: 100%
   - **AUC**: 0.9920

## Results

The models were evaluated on the following metrics:

- **Accuracy**: Measures the overall correctness of the model.
- **Sensitivity (Recall)**: Measures the ability of the model to correctly classify spam messages.
- **Specificity**: Measures the ability of the model to correctly classify non-spam messages.
- **AUC**: The area under the ROC curve, a measure of the model’s discriminatory power.

## Best Model

The **Multinomial Naive Bayes** model was identified as the best model, with an accuracy of 98%, sensitivity of 90.5%, specificity of 99.2%, and an AUC of 0.9836. This model provides a good balance between correctly identifying spam and non-spam messages.


