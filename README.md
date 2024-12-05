# Spam Email Classification using NLP and Machine Learning

## Project Overview

This project focuses on classifying emails as **Spam** (unsolicited emails) or **Ham** (legitimate emails) using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The project uses the **spam.csv** dataset to train a model, which is then used for classifying new emails as spam or ham.

## Project Structure

- **`Spam Detector.ipynb`**: Jupyter notebook for exploring the dataset, data preprocessing, model training, and evaluation.
- **`spam.csv`**: The dataset containing labeled emails (Spam and Ham).
- **`spam.pkl`**: Serialized machine learning model that has been trained on the dataset.
- **`spamDetector.py`**: Python script that contains the logic for loading the trained model and making predictions on new email data.
- **`vec.pkl`**: Serialized vectorizer used to transform the text data (email content) into numerical features before feeding it to the model.

## Key Features

1. **Email Preprocessing**: Text cleaning, tokenization, stopword removal, and stemming/lemmatization.
2. **Feature Extraction**: Using **TF-IDF** (Term Frequency-Inverse Document Frequency) and other NLP techniques to extract meaningful features from email content.
3. **Model Training**: Training machine learning models like **Naive Bayes**, **Logistic Regression**, and other classifiers for detecting spam emails.
4. **Spam Classification**: Classifying emails as either *Spam* or *Ham* based on trained models.
5. **Model Prediction**: Using the saved model (`spam.pkl`) to classify new email data by running the `spamDetector.py` script.

## How to Use

1. Clone or download this repository.
2. Open **`Spam Detector.ipynb`** in Jupyter Notebook and run the cells to explore the dataset, train the model, and evaluate its performance.
3. Alternatively, you can use the **`spamDetector.py`** script to make predictions by loading the pre-trained model and vectorizer. Ensure that the **`spam.pkl`** and **`vec.pkl`** files are present in the same directory as the script.
4. To predict the spam status of a new email, load the model and vectorizer, then pass the email content to the prediction function.

## Models and Techniques Used

- **Naive Bayes**: A probabilistic classifier that works well with text data.
- **Support Vector Machine (SVM)**: A powerful classifier for high-dimensional data.
- **Logistic Regression**: A binary classification model.
- **LSTM (Long Short-Term Memory)**: A deep learning model for handling sequential data like email content.

## Results and Performance

The model achieves an accuracy of around **95%** on the test dataset, with good precision, recall, and F1-score values for spam detection. 

## Future Improvements

- **Advanced NLP Models**: Incorporating state-of-the-art models like **BERT** for better text classification performance.
- **Real-Time Spam Filtering**: Integrating the spam classification model into an email service or real-time email client.
- **Hybrid Models**: Combining machine learning and deep learning techniques for improved accuracy.

