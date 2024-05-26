#pandas and numpy for df manipulation
import pandas as pd
import numpy as np
import re
import nltk
import statistics
import random

#Preprocessing: tokenization and lemmatization, and stopwords removal
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
sent_tokenizer = PunktSentenceTokenizer()
#nltk.download('stopwords')
from nltk.corpus import stopwords

from textblob import TextBlob

#Regression Metrics
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

#Vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizer = TfidfVectorizer()
bow_vectorizer = CountVectorizer()

#Classification and Metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
#############################################################################
stop_words = set(stopwords.words('english'))

def stopword_remover(tokenized_comment):
    """
    Removes stop words from a tokenized comment.

    Args:
    - tokenized_comment (list): A list of tokens representing a comment.

    Returns:
    - clean_comment (list): A list of tokens with stop words removed.
    """
    clean_comment = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_comment.append(token)
    return clean_comment
#############################################################################
def preprocessor(raw_text,
                 lowercase=True,
                 leave_punctuation=False,
                 remove_stopwords=True,
                 correct_spelling=False,
                 lemmatization=False,
                 tokenized_output=False,
                 sentence_output=False
                 ):
    """
    Preprocesses raw text based on specified operations.

    Args:
    - raw_text (str): The input text to be preprocessed.
    - lowercase (bool): Whether to convert text to lowercase (default: True).
    - leave_punctuation (bool): Whether to retain punctuation (default: False).
    - remove_stopwords (bool): Whether to remove stop words (default: True).
    - correct_spelling (bool): Whether to perform spelling correction (default: False).
    - lemmatization (bool): Whether to lemmatize words (default: False).
    - tokenized_output (bool): Whether to output tokenized text (default: False).
    - sentence_output (bool): Whether to output sentences (default: False).

    Returns:
    - clean_text (str or list): The preprocessed text based on the specified operations.
    """
    # Convert to lowercase if specified
    if lowercase:
        clean_text = raw_text.lower()
    else:
        clean_text = raw_text
    
    # Remove newline characters
    clean_text = re.sub(r'(\**\\[nrt]|</ul>)', ' ', clean_text)
    # Remove URL
    clean_text = re.sub(r'(\bhttp[^\s]+\b)', ' ', clean_text)
    # Remove isolated consonants
    clean_text = re.sub(r'\b([^aeiou-])\b', ' ', clean_text)
    
    # Remove punctuation if specified
    if not leave_punctuation:
        clean_text = re.sub(r'([\.\,\;\?\!\:\(\)])|(\b\(\b)|(\b\)\b)', ' ', clean_text)
        clean_text = re.sub(r'\(', ' ', clean_text)
        clean_text = re.sub(r'\)', ' ', clean_text)
        clean_text = re.sub(r'[`\'"*]', '', clean_text)
    
    # Correct spelling if specified
    if correct_spelling:
        incorrect_text = TextBlob(clean_text)
        clean_text = incorrect_text.correct()
    
    # Tokenize
    clean_text = word_tokenize(str(clean_text))
    
    # Remove stopwords if specified
    if remove_stopwords:
        clean_text = stopword_remover(clean_text)
    
    # Lemmatize if specified
    if lemmatization:
        for pos_tag in ["v", "n", "a"]:
            clean_text = [lemmatizer.lemmatize(token, pos=pos_tag) for token in clean_text]

    # Re-join tokens if tokenized output is not requested
    if not tokenized_output:
        clean_text = " ".join(clean_text)
        # Remove space before punctuation
        clean_text = re.sub(r'(\s)(?!\w)', '', clean_text)
    
    # Output sentences if specified
    if sentence_output:
        # Split into sentences
        clean_text = sent_tokenizer.tokenize(str(clean_text))

    return clean_text

#############################################################################

def word_cloud_generator(dataset, class_s, test_name, column_for_corpus):
    """
    Generates word clouds for different classes in a dataset.

    Args:
    - dataset (DataFrame): The dataset containing text and labels.
    - class_s (list): List of classes or tags to generate word clouds for.
    - test_name (str): Name of the test or experiment.
    - column_for_corpus (str): Name of the column in the dataset containing text corpus.

    Returns:
    - dataset (DataFrame): The modified dataset (columns might have been added temporarily).

    Generates word clouds for each class specified in class_s based on the given dataset.
    Utilizes two methods: Bag of Words (BoW) and TF-IDF, generating word clouds for each class
    using each method separately.
    """

    methods = ["bow", "tfidf"]
    num_methods = len(methods)
    num_classes = len(class_s)
    fig, axes = plt.subplots(num_classes, num_methods, figsize=(18, 15))
    
    # Iterate through the methods (BoW and TF-IDF)
    for i, method in enumerate(methods):
        # Iterate through each class in the class_s list
        for j, class_ in enumerate(class_s):
            # Create a dummy column in the dataset based on class presence
            dataset[f"has_class_{class_}"] = dataset["tag"].apply(lambda tag: 1 if class_ in tag else 0)
            # Extract the corpus for the current class
            corpus = list(dataset.loc[dataset[f"has_class_{class_}"] == 1, column_for_corpus])

            # Choose the vectorizer based on the method (BoW or TF-IDF)
            if method == "bow":
                vectorizer = CountVectorizer()
            else:
                vectorizer = TfidfVectorizer()
            # Fit the vectorizer and transform the corpus
            fitted_model = vectorizer.fit_transform(corpus)
            # Get the top frequencies for the words
            top_freqs = dict(zip(vectorizer.get_feature_names_out(), np.ravel(fitted_model.sum(axis=0)).tolist()))

            # Generate word cloud based on the top word frequencies
            wc = WordCloud(background_color="white", max_words=120, width=220, height=220)
            wc.generate_from_frequencies(top_freqs)

            # Plot word cloud in the appropriate subplot
            axes[j, i].imshow(wc, interpolation='bilinear')
            axes[j, i].axis("off")
            axes[j, i].set_title(f"Word Cloud - {class_} ({method})")

            # Save word cloud as an image
            wc.to_file(f"wc_{test_name}_{method}_{class_}.png")

            # Drop the dummy column from the dataset
            dataset = dataset.drop(columns=[f"has_class_{class_}"])

    # Adjust subplot layout and display the word clouds
    plt.tight_layout()
    plt.show()
    
    return dataset

#############################################################################
def plot_confusion_matrix(y_test_decoded, y_pred_decoded):
    """
    Plots a confusion matrix based on predicted and true labels.

    Args:
    - y_test_decoded (array-like): True labels.
    - y_pred_decoded (array-like): Predicted labels.

    Returns:
    - None
    """
    unique_labels = sorted(list(set(y_test_decoded) | set(y_pred_decoded)))  # Get unique labels
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)  # Calculate confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Calculate percentages

    # Plotting confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, cmap='mako_r', fmt='.2f',
                xticklabels=list(unique_labels),
                yticklabels=list(unique_labels))

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%)')
    plt.show()
    return

#############################################################################
# Function for Stratified K-Fold Cross-Validation
def stratified_kfold_crossval(songs, data, classifier, num_folds):
    """
    Conducts stratified k-fold cross-validation using a specified classifier on text data.

    Args:
    - data (list or array-like): Text data for classification.
    - classifier (object): The classifier to be trained and evaluated.
    - num_folds (int): Number of folds for stratified k-fold cross-validation.

    Returns:
    - tuple: A tuple containing average evaluation scores (accuracy, precision, recall, F1-score) across folds,
             and the trained model.
    """
    
    # Initialize Stratified K-Fold object
    stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    # Initialize Label Encoder
    encoder = LabelEncoder()

    # Transform text data into numerical format using TF-IDF Vectorizer
    corpus = data
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(corpus)

    # Prepare target labels for classification
    y = np.ravel(songs["tag"]) 
    encoder.fit(songs[["tag"]])
    y_encoded = encoder.transform(songs[["tag"]])

    # Initialize lists to store evaluation metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    fold = 1

    # Iterate through folds using Stratified K-Fold
    for train_index, test_index in stratified_kfold.split(X, y):
        # Split data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Train the provided classifier on the training data
        model = classifier
        model.fit(X_train, y_train)

        # Generate predictions on the test set
        y_pred = model.predict(X_test)

        # Compute evaluation metrics for this fold
        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')

        # Append metrics to respective lists
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(recall)
        f1_scores.append(f1)
        print("Fold:", fold)  #Since this is something that takes a while to run, we place this print here as a checkpoint

        fold += 1

    # Calculate average scores across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    # Return average scores and the trained model
    return avg_accuracy, avg_precision, avg_recall, avg_f1, model, tfidf_vectorizer