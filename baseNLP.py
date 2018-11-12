import pandas as pd

df_yelp = pd.read_table('C:/Users/prana/Desktop/SDS/sentiment labelled sentences/yelp_labelled.txt')

df_imdb = pd.read_table('C:/Users/prana/Desktop/SDS/sentiment labelled sentences/imdb_labelled.txt')

df_amz = pd.read_table('C:/Users/prana/Desktop/SDS/sentiment labelled sentences/amazon_cells_labelled.txt')

frames = [df_yelp,df_imdb,df_amz]

for colname in frames:
    colname.columns = ['Message','Target']


#df_imdb.columns
for colname in frames:
    print(colname.columns)
    
keys = ['Yelp','IMDB','Amazon']

df = pd.concat(frames,keys=keys)

df.shape

df.head()

df.to_csv('C:/Users/prana/Desktop/SDS/sentimentdataset1.csv')

df.columns

df.isnull().sum()

#conda install -c conda-forge spacy=1.8.2
#python -m spacy download en

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en')

stopwords = list(STOP_WORDS)

docx = nlp("This is how John Walker was walking. He was also running beside the lawn")

for word in docx:
    print(word.text,"Lemma =>",word.lemma_)

for word in docx:
    if word.lemma_ != "-PRON-":
        print(word.lemma_.lower().strip())


[word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in docx]

for word in docx:
    if word.is_stop == False and not word.ispunct:
        print(word)
        
[word for word in docx if word.is_stop == False and not word.is_punct]


import string
punctuations = string.punctuation

from spacy.lang.en import English
parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    return mytokens



from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class predictors(TransformerMixin):
    def transform(self,X,**transform_params):
        return[clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self,deep=True):
        return{}

def clean_text(text):
    return text.strip().lower()

vectorizer = Countvectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
classifier = LinearSVC()

tfidvectorizer = TfidVectorizer(tokenizer = spacy_tokenizer)

from sklearn.model_selection import train_test_split

X = df['Message']
ylabels = df['Target']


X_train, X_test, y_train, y_test = train_test_split(X,ylabels,test_size=0.2,)
        

pipe = Pipeline([("cleaner",predictors()),
                 ('vectorizer',vectorizer),
                 ('classifier',classifier)])
    
pipe.fit(X_train,y_train)

sample_prediction = pipe.predict(X_test)

for(sample,pred) in zip(X_test,sample_prediction):
    print(sample,"Predicition=>",pred)


print("Accuracy: ", pipe.score(X_test,y_test))
print("Accuracy: ", pipe.score(X_test,sample_prediction))


print("Accuracy: ", pipe.score(X_train,y_train))

pipe.predict(["This was a great movie"])

example = ["I do enjoy my job",
           "What a poor product!, I will have to get a new one",
           "I feel amazing!"]

pipe.predict(example)














'''
mysents = []

for i in df_clean.Message:
    docx = nlp(i)
    procs = [word for word in docx if word.is_stop != True and not word.is]
'''