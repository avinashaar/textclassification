

import sys
import nltk
import sklearn 
import numpy as np
import pandas as pd
# Load data

df = pd.read_table('SMSSpamCollection',header = None,encoding = 'utf-8')
print(df.info())

classes = df[0]
print(classes.value_counts())

#preprocess the data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(classes)
print(classes[:10])
print(y[:10]) 

test_messages = df[1]
print(test_messages[:10])

processed = test_messages.str.replace(r'^,+@[^\.].*\.[a-z]{z,}$','emailadder')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
processed = processed.str.replace(r'£/\$','moneysymb')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonumbr')
processed = processed.str.replace(r'\d+(\.\d+)?','numbr')

processed = processed.str.replace(r'[^\w\d\s]',' ')
rocessed = processed.str.replace(r'\s+','')
rocessed = processed.str.replace(r'^\s+/\s+?$','')
rocessed = processed.str.lower()
print(processed)

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x:''.join(term for term in x.split() if term not in stop_words))
 
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x:''.join(ps.stem(term) for term in x.split()))
print(processed)

from nltk.tokenize import word_tokenize
nltk.download('punkt')
all_words = []
for message in processed:
        words = word_tokenize(message)
for w in words:
        all_words.append(w)
all_words = nltk.FreqDist(all_words)

print('Number of words:{}'.format(len(all_words)))
print('Most common words:{}'.format(all_words.most_common(500)))

word_features = list(all_words.keys())[ :1500]
def find_features(message):
        words = word_tokenize(message)
        features = {}
        for word in word_features:
                features[word] = (word in words)
                
        return features

features = find_features(processed[0])
for key, value in features.items():
        if value == True:
                print key

messages = zip(processed,y)
seed = 1
np.random.seed = seed
np.random.shuffle(messages)
featuresets = [(find_features(test),label) for (test , label) in messages]
from sklearn import model_selection
training , testing = model_selection.train_test_split(featuresets , test_size = 0.25, random_state = seed)
print('training:{}'.format(len(training)))
print('testing:{}'.format(len(testing)))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

names = ['K Nearest neighbores','Decision Tree','Random Forest','Logistic Regression','SGD classifier','Navies Bayes','SVM linear']
classifier = [
             KNeighborsClassifier(),
             DecisionTreeClassifier(),
             RandomForestClassifier(),
             LogisticRegression(),
             SGDClassifier(max_iter = 100),
             MultinomialNB(),
             SVC(kernel = 'linear')
             ]

models = zip(names, classifier)
print(models)

from nltk.classify.scikitlearn import SklearnClassifier
for name ,model in models:
        nltk_model = SklearnClassifier(model)
        nltk_model.train(training)
        accuracy = nltk.classify.accuracy(nltk_model,testing ) * 100
        print('{}: Accuracy:{}'.format(name,accuracy))
        


