import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Remove ellipsis
    text = re.sub(r"\.{3}", " ", text)
    # Replace consecutive digits with spaces
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Replace "--" and "”" with spaces
    text = text.replace("--", " ")
    text = text.replace("”", " ")
    return text

def tokenize_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word, pos='v') for word in text]
    return text

    

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv("grouped_traumas_miscarriage_variables.csv")
    
    # Apply preprocessing to the text column
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Tokenize the cleaned text
    stop_words = set(stopwords.words('english'))
    df['tokenized_text'] = df['clean_text'].apply(word_tokenize)
    df['tokenized_text'] = df['tokenized_text'].apply(lambda words: [word for word in words if word not in stop_words])
    #lambda functions = small inline functions "lambda arguments: expression"

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    df["clean_text"] = df["tokenized_text"].apply(lambda tokens: [lemmatizer.lemmatize(token, pos='v') for token in tokens])

    # Save the preprocessed DataFrame to a CSV file
    df.to_csv("miscarriage_preprocessed_test.csv", index=False)