from  numpy import *
import pandas as pd
from IPython.display import display
from keras.utils import to_categorical
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup
import re
from tqdm import tqdm
class movie():
    def __init__(self):
        self.trainpath = '/Users/zuobangbang/Desktop/movie/train.tsv'
        self.testpath = '/Users/zuobangbang/Desktop/movie/test.tsv'
        self.submit = '/Users/zuobangbang/Desktop/movie/sampleSubmission.csv'

    def read_data(self):
        train=pd.read_csv(self.trainpath,sep='\t')
        display(train.head())
        print(train.shape)
        test=pd.read_csv(self.testpath,sep='\t')
        print(test.head())
        train=self.clean_sentences(train)
        return train

    def clean_sentences(self,df):
        lemmatizer = WordNetLemmatizer()
        reviews = []

        for sent in tqdm(df['Phrase']):
            # remove html content
            review_text = BeautifulSoup(sent).get_text()

            # remove non-alphabetic characters
            review_text = re.sub("[^a-zA-Z]", " ", review_text)

            # tokenize the sentences
            words = word_tokenize(review_text.lower())

            # lemmatize each word to its lemma
            lemma_words = [lemmatizer.lemmatize(i) for i in words]

            reviews.append(lemma_words)

        return (reviews)

    def main_code(self):
        self.read_data()


if __name__=='__main__':
    m=movie()
    m.main_code()