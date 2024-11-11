# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
import string
import requests
from bs4 import BeautifulSoup as bs4
import csv
import time

# Load dataset from CSV file into a DataFrame
data_file = 'la-labelled-data.csv'
X = pd.read_csv(data_file, sep=",", usecols=range(1, 26))

# Separate majority and minority classes for balancing the dataset
df_majority = X[X.SPAM == 0]  # Majority class (not spam)
df_minority = X[X.SPAM == 1]   # Minority class (spam)

# Upsample minority class to match majority class size for balanced training data
df_minority_upsampled = resample(df_minority, replace=True, n_samples=1000, random_state=123)
df_majority_downsampled = resample(df_majority, replace=False, n_samples=1000, random_state=123)

# Combine majority class with upsampled minority class to create a balanced dataset
df_upsampled = pd.concat([df_majority_downsampled, df_minority_upsampled])
print("New class counts after upsampling:")
print(df_upsampled.SPAM.value_counts())

# Assign the upsampled DataFrame back to X for further processing
X = df_upsampled

# Preprocess the 'description' column by removing non-ASCII characters and punctuation
X['description'].replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
exclude = set(string.punctuation)

def remove_punctuation(text):
    """Remove punctuation from text."""
    try:
        return ''.join(char for char in text if char not in exclude)
    except Exception as e:
        print(f"Error in removing punctuation: {e}")
        return text

# Apply the punctuation removal function to the 'description' column
X['description'] = X['description'].apply(remove_punctuation)

# Create a new binary column indicating if a post has been reposted or not
X['reposted'] = np.where(X['post_id_old'].isnull(), 0, 1)

# Fill missing values in numerical columns with zeros for consistency in processing
X['square_footage'] = X['square_footage'].fillna(0).astype(int)
X['price'] = X['price'].fillna(0)

# Remove any rows where square footage is labeled as 'loft'
X = X[~(X['square_footage'] == 'loft')]

# Ensure that square footage is of integer type after cleaning data
X['square_footage'] = X['square_footage'].astype(int)

# Fill missing descriptions with an empty string to avoid errors during vectorization
X['description'] = X['description'].fillna('')

# Pre-process categorical data by converting it into binary variables using one-hot encoding
X_cat = pd.get_dummies(X, columns=['reposted', 'laundry', 'parking', 'cat', 'dog', 'smoking', 'furnished', 'borough', 'housing_type'])

# Drop irrelevant columns from the categorical DataFrame to keep only necessary features for modeling
columns_to_drop = ['post_date', 'post_time', 'update_date', 'update_time', 
                   'title', 'price', 'images', 'square_footage', 
                   'bedroom', 'bathroom', 'post_id_old', 
                   'post_id', 'parenthesis', 'description', 
                   'URL', 'Reason']
X_cat.drop(columns=columns_to_drop, inplace=True)

# Define the target variable (labels) for classification tasks
y = X['SPAM']

print("Shape of feature set:", X.shape)
print("Shape of target variable:", y.shape)

# Prepare numerical features by selecting relevant columns from the DataFrame
X_num = pd.DataFrame(X, columns=['price', 'bathroom', 'bedroom', 'images', 'square_footage'])

# Prepare text data for TF-IDF vectorization to convert text into numerical format suitable for modeling
X_text = X['description']

# Initialize TF-IDF Vectorizer with English stop words removal for text processing
tvf = TfidfVectorizer(stop_words='english')
X_text_vectorized = tvf.fit_transform(X_text)

# Fill missing values in numerical features with zeros to ensure no NaN values remain before model training
X_num.fillna(0, inplace=True)

# Combine all processed features into a single sparse matrix for model training
from scipy import sparse

X_combined = sparse.hstack((X_text_vectorized, X_cat, X_num)).toarray()

# Feature selection using Random Forest Classifier to reduce dimensionality of feature set based on importance scores
clf_feature_selection = RandomForestClassifier(n_estimators=100)
clf_feature_selection.fit(X_combined, y.ravel())
sfm = SelectFromModel(clf_feature_selection, threshold=0.001)
X_selected_features = sfm.fit_transform(X_combined, y.ravel())

# Split the dataset into training and testing sets for model evaluation (70% training and 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size=0.3)

# Initialize various classifiers to be used in ensemble methods later on
clf1 = RandomForestClassifier(n_estimators=100)
clf2 = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf3 = ExtraTreesClassifier(n_estimators=100)
clf4 = ExtraTreesClassifier(n_estimators=100, criterion='entropy')
clf5 = GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
clf6 = DecisionTreeClassifier()
clf7 = svm.SVC(gamma=0.001, C=100)
clf8 = KNeighborsClassifier()
clf9 = GaussianNB()

# Assemble classifiers into a list for ensemble voting classifier setup
predictors = [('RF_gini', clf1), ('RF_entropy', clf2), ('ET_gini', clf3), 
              ('ET_entropy', clf4), ('GB', clf5), ('DT', clf6), 
              ('SVM', clf7), ('KNN', clf8), ('NB', clf9)]

# Create a Voting Classifier that combines predictions from multiple classifiers to improve accuracy and robustness
voting_classifier = VotingClassifier(estimators=predictors)

# Fit the voting classifier on the training data and evaluate its performance on the test set
voting_classifier.fit(X_train, y_train)
predicted_labels_voting_classifier = voting_classifier.predict(X_test)

# Print confusion matrix and accuracy score of the voting classifier's predictions against actual labels in test set
print("Confusion Matrix for Voting Classifier:")
print(confusion_matrix(predicted_labels_voting_classifier, y_test))
print("Accuracy Score of Voting Classifier:", accuracy_score(predicted_labels_voting_classifier, y_test))

# Initialize Stacking Classifier using previously defined classifiers and logistic regression as meta-classifier for final predictions.
from mlxtend.classifier import StackingClassifier

stacking_classifier = StackingClassifier(classifiers=predictors[:-1], meta_classifier=GaussianNB())
stacking_classifier.fit(X_train, y_train)

# Make predictions using stacking classifier and evaluate its performance on test data.
predicted_labels_stacking_classifier = stacking_classifier.predict(X_test)
print("Confusion Matrix for Stacking Classifier:")
print(confusion_matrix(predicted_labels_stacking_classifier, y_test))
print("Accuracy Score of Stacking Classifier:", accuracy_score(predicted_labels_stacking_classifier, y_test))
