import numpy as np
import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Define column names for the dataset
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']

# Load dataset and replace target value 4 with 1 for positive tweets
twitter_data = pd.read_csv("C:\\Users\\fanne\\my_kaggle_dataset\\sentiment140.csv", 
                           names=column_names, 
                           encoding="ISO-8859-1")
twitter_data.replace({'target': {4: 1}}, inplace=True)
print(twitter_data['target'].value_counts())

# Initialize the PorterStemmer
port_stem = PorterStemmer()

# Original stemming function (uncommented and renamed for clarity)
def stemming_original(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    stemmed_content = stemmed_content.lower()            # Convert to lowercase
    stemmed_content = stemmed_content.split()            # Tokenize
    stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                       if word not in stopwords.words('english')]  # Remove stopwords and stem
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Faster text preprocessing function (currently in use)
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    content = content.lower().split()             # Convert to lowercase and tokenize
    content = [port_stem.stem(word) for word in content 
               if word not in stopwords.words('english')]
    return ' '.join(content)

# Apply stemming on a random sample of 500 tweets for efficiency.
twitter_data_sample = twitter_data.sample(n=500, random_state=42)
# Choose which stemming function to use:
# For the original function, use: stemming_original
twitter_data_sample['stemmed_content'] = twitter_data_sample['text'].apply(stemming)

# Extract features and target
x = twitter_data_sample['stemmed_content'].values
y = twitter_data_sample['target'].values

# Print first 5 processed tweets and target values
print("First 5 processed tweets:", x[:5])
print("First 5 target values:", y[:5])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print("Shape of data:", x.shape, x_train.shape, x_test.shape)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate the model on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy score on the training data:", training_data_accuracy)

# Evaluate the model on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy score on the test data:", test_data_accuracy)

# Save the trained model to disk
filename = "trained_model.sav"
pickle.dump(model, open(filename, 'wb'))

# Load the model and make a prediction on a single sample
loaded_model = pickle.load(open(filename, "rb"))
# Select a sample tweet from the test set (e.g., index 20)
x_new = x_test[20]
x_new = x_new.toarray()  # Convert sparse matrix to dense array

prediction = loaded_model.predict(x_new)
print("Actual Label:", y_test[20])
print("Predicted Label:", prediction[0])

# Print prediction result
if prediction[0] == 0:
    print("Negative Tweet")
else:
    print("Positive Tweet")
