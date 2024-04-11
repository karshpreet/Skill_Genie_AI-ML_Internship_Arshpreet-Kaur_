import pandas as pd 

import re 

import nltk 

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk.stem import PorterStemmer 

from sklearn.feature_extraction.text import TfidfVectorizer 

 

# Download NLTK data 

nltk.download('punkt') 

nltk.download('stopwords') 

 

# Load the dataset 

data = pd.read_csv("C:/Users/Asus/Downloads/emails.csv", encoding='latin-1') 

 

# Explore the structure and characteristics of the dataset 

print(data.head()) 

print(data.info()) 

 

# Preprocess the email text 

def preprocess_text(text): 

    if isinstance(text, str): 

        # Remove special characters and numbers 

        text = re.sub(r'[^a-zA-Z]', ' ', text) 

        # Convert to lowercase 

        text = text.lower() 

    else: 

        text = ''  # Set to empty string if not a string 

    return text 

 

# Apply preprocessing to the email text column 

data['text'] = data['text'].apply(preprocess_text) 

 

# Tokenization, stemming, and vectorization 

stop_words = set(stopwords.words('english')) 

stemmer = PorterStemmer() 

 

def tokenize_and_stem(text): 

    tokens = word_tokenize(text) 

    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words] 

    return tokens 

 

# Check for empty documents after preprocessing 

data['text'] = data['text'].apply(lambda x: x if len(x) > 0 else 'empty') 

 

# Vectorize the text using TF-IDF 

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, max_features=5000, token_pattern=None) 

X = tfidf_vectorizer.fit_transform(data['text']) 
