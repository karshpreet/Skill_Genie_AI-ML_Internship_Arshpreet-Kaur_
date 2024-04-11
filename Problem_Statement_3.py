import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load IMDb movie reviews dataset
# Assuming the dataset is stored in a CSV file with 'review' and 'sentiment' columns
df = pd.read_csv('IMDB Dataset.csv')

# Explore dataset
print("Dataset Info:")
print(df.info())
print("\nDataset Sample:")
print(df.head())

# Distribution of positive and negative reviews
print("\nDistribution of Sentiments:")
print(df['sentiment'].value_counts())

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    # porter = PorterStemmer()
    # stemmed_tokens = [porter.stem(word) for word in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

# Apply preprocessing to all reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Display cleaned reviews
print("\nCleaned Reviews:")
print(df['cleaned_review'].head())
