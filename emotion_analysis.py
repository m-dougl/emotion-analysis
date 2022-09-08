
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from unicodedata import normalize

import pandas as pd
import nltk
import numpy as np
import string
import emotion_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
nltk.download('rslp')
nltk.download('stopwords')


def open_dataset(dataset_name, dataset_type):
    if dataset_type == 'excel' or 'xlsx':
        df = pd.read_excel(str(dataset_name))
        return df
    elif dataset_type == 'csv':
        df = pd.read_excel(str(dataset_name))
        return df

def remove_characters(txt):
    sc = [k for k in txt.lower() if k not in string.punctuation]
    sc_ = ''.join(sc)
    return sc_

def remove_accents(txt):
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

def tokenize(txt):
    return RegexpTokenizer('\w+').tokenize(txt)

def untokenize(txt):
    return (' ').join(i for i in txt if len(i)>1)

def remove_stopwords(txt):
    return [w for w in txt if w not in nltk.corpus.stopwords.words('portuguese')]

def stemming(txt):
    return (' ').join([nltk.stem.RSLPStemmer().stem(i) for i in txt.split()])

def simple_train(classifier_name, X, y):
    if classifier_name == 'NB':
        clf = MultinomialNB().fit(X, y)
        return clf
    elif classifier_name == 'SVM':
        clf = SVC(gamma=1, C=1, kernel = 'rbf').fit(X, y)
        return clf
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = 3).fit(X, y)
        return clf

def cv_train(classifier_name, X, y, n_fold=5):
    if classifier_name == 'NB':
        clf = GridSearchCV(estimator  = MultinomialNB(),
                           param_grid = {'alpha': [0.001,0.01,0.1,1,10,100,1000],
                                         'fit_prior':[True, False]},
                           cv         = n_fold).fit(X, y)
        print(10*'==')
        print(f'Naive Bayes best parameters: {clf.best_params_}')
        print(f'Naive Bayes best accuracy in {n_fold} folds: {clf.best_score_*100}')
        print(10*'==')
        return clf
    
    elif classifier_name == 'SVM':    
        clf = GridSearchCV(estimator  = SVC(),
                           param_grid = {'C': [0.001,0.01,0.1,1,10,100,1000],
                                         'kernel': ['linear','poly','rbf','sigmoid'],
                                         'gamma': [0.001,0.01,0.1,1,10,100]},
                           cv         = n_fold).fit(X, y)
        print(10*'==')
        print(f'Support Vector Machine best parameters: {clf.best_params_}')
        print(f'Support Vector Machine best accuracy in {n_fold} folds: {clf.best_score_*100}')
        print(10*'==')
        return clf
    
    elif classifier_name == 'KNN':
        clf = GridSearchCV(estimator  = KNeighborsClassifier(),
                           param_grid = {'n_neighbors': [2,3,4,5,6],
                                         'algorithm':['auto','ball_tree','kd_tree','brute']},
                           cv         = n_fold).fit(X, y)
        print(10*'==')
        print(f'K-Nearest Neighbors best parameters: {clf.best_params_}')
        print(f'K-Nearest Neighbors best accuracy in {n_fold} folds: {clf.best_score_*100}')
        print(10*'==')    
        return clf

def vectorizer(X, vec_type):
    if vec_type == 'countvectorizer':
        CountVec = CountVectorizer()
        numeric_matrix = CountVec.fit_transform(X)
        return numeric_matrix
    
    elif vec_type == 'tfidf-vectorizer' or 'tfidf':
        Tfidf = TfidfVectorizer()
        numeric_matrix = Tfidf.fit_transform(X)
        return numeric_matrix
    
def dataset_split(X, y, train_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       train_size   = train_size,
                                                       stratify       = y,
                                                       random_state = 0,
                                                       shuffle      = True)
    return X_train, X_test, y_train, y_test

def confusion_matrix_plot(list_predict, models_names, y_true):
    cm_list = []
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(list_predict)):
        cm_list.append(confusion_matrix(y_true, list_predict[i]))
    for i in range(len(cm_list)):
        cm = cm_list[i]
        model = models_names[i]
        subplots = fig.add_subplot(2, 3, i+1).set_title(model)
        cm_plot = sns.heatmap(cm, annot=True, cmap='Reds')
        cm_plot.set_xlabel('')
        cm_plot.set_ylabel('')
        
def emotion_plot(y_train, y_test, labels):
    set_ = {'Train': y_train,
            'Test': y_test}
    train, test = set_['Train'].value_counts(), set_['Test'].value_counts()
    width = .35
    fig, axis = plt.subplots(figsize=(10, 6))
    plt.title(f'Train and test split with {len(y_train)+len(y_test)} comments')
    rec1 = axis.bar(np.arange(len(labels)) - width/2, train, width, label = 'Train')
    rec2 = axis.bar(np.arange(len(labels)) + width/2, test, width,  label = 'Test')
    axis.bar_label(rec1, padding=3)
    axis.bar_label(rec2, padding=3)
    plt.xticks(np.arange(len(labels)), labels)
    plt.legend()
    plt.show()

def metrics_evaluation(models_names, list_predict, y_true):
    accuracy    = []
    precision   = []
    recall      = []
    f_score    = []
    for i in range(len(models_names)):
        accuracy.append(int(accuracy_score(y_true, list_predict[i])*100))
        precision.append(int(precision_score(y_true, list_predict[i], average='macro')*100))
        recall.append(int(recall_score(y_true, list_predict[i], average='macro')*100))
        f_score.append(int(f1_score(y_true, list_predict[i], average = 'macro')*100))
    
    fig, axis = plt.subplots(2, 2, figsize= (15,10), sharey= True)
    axis[0,0].set_ylim([0, 100])
    axis[0,0].set_ylabel('Accuracy(%)')
    axis[0,0].bar(models_names, accuracy)
    for i in axis[0,0].patches:
        axis[0,0].annotate(i.get_height(),
                           (i.get_x()+i.get_width()/2, i.get_height()),
                           ha='center', va='baseline', fontsize='10',
                           xytext=(0,1), textcoords='offset points')
    
    axis[0,1].set_ylabel('Precision(%)')
    axis[0,1].bar(models_names, precision)
    for i in axis[0,1].patches:
        axis[0,1].annotate(i.get_height(),
                           (i.get_x()+i.get_width()/2, i.get_height()),
                           ha='center', va='baseline', fontsize='10',
                           xytext=(0,1), textcoords='offset points')
        
    axis[1,0].set_ylabel('Recall(%)')
    axis[1,0].bar(models_names, recall)
    for i in axis[1,0].patches:
        axis[1,0].annotate(i.get_height(),
                           (i.get_x()+i.get_width()/2, i.get_height()),
                           ha='center', va='baseline', fontsize='10',
                           xytext=(0,1), textcoords='offset points')
    
    axis[1,1].set_ylabel('F1-Scores(%)')   
    axis[1,1].bar(models_names, f_score)
    for i in axis[1,1].patches:
        axis[1,1].annotate(i.get_height(),
                           (i.get_x()+i.get_width()/2, i.get_height()),
                           ha='center', va='baseline', fontsize='10',
                           xytext=(0,1), textcoords='offset points')