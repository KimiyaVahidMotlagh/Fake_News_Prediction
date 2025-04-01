# Fake News Detection

## Artificial Intelligence for Fake News Detection
This project utilizes machine learning models to classify news articles as either real or fake. It applies Natural Language Processing (NLP) techniques to preprocess text, extract features, and train multiple classification models to achieve high accuracy.

## Table of Contents
1. [Dataset and Preprocessing](#dataset-and-preprocessing)
2. [Feature Engineering](#feature-engineering)
3. [Data Visualization](#data-visualization)
4. [Model Training](#model-training)
5. [Evaluation and Performance](#evaluation-and-performance)
6. [Prediction on New Data](#prediction-on-new-data)

---

## Dataset and Preprocessing
The dataset used (`final_en.csv`) consists of news articles labeled as real (0) or fake (1). To enhance accuracy, preprocessing includes:
- Converting text to lowercase
- Tokenization and lemmatization using spaCy
- Removing stop words, common, and rare words

The text features are extracted from both the `title` and `text` columns to form a processed dataset.

## Feature Engineering
After preprocessing, we use the **TF-IDF Vectorizer** to transform text into numerical vectors. Key steps include:
- Setting n-gram range (1,2)
- Selecting a maximum of 5000 features
- Removing English stop words

## Data Visualization
To gain insights into the dataset, we use:
- **Seaborn count plot** to visualize label distribution
- **Word clouds** to show common words in fake and real news articles

## Model Training
We train multiple machine learning models to compare performance:
- **Logistic Regression**
- **Naïve Bayes**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **Decision Tree**
- **AdaBoost**
- **LightGBM**
- **Multilayer Perceptron (MLP Neural Network)**

Each model is trained on 80% of the dataset and tested on the remaining 20%.

## Evaluation and Performance
The models are evaluated based on **accuracy** and a **classification report**. The results are displayed in descending order of performance to determine the most effective model.

## Prediction on New Data
A function `predict_news()` allows predictions on new articles. Example usage:
```python
new_news = ["Breaking news: Stock market crashes due to global tension!",
            "Scientists discover a new method to reverse climate change."]
predictions = predict_news(models, vectorizer, new_news)
for model, pred in predictions.items():
    print(f"{model} Predictions: {pred}")
```

This project provides a solid foundation for detecting fake news using AI models. Further enhancements could involve deep learning techniques or incorporating external fact-checking sources for improved accuracy.
