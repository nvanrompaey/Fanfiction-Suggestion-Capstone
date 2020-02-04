import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
stop_words = set(stopwords.words('english'))


def POSparser(stringinq):
    #Takes a string as a name you want the new row to be called, 
    # and a dataframe + column on which to perform the transformation
    tokenized = sent_tokenize(stringinq)
    Taglist = []
    for i in tokenized:
        # DO THIS: minimize words and add dummy column
        wordsList = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(wordsList)
        Taglist.append([i[1] for i in tagged])
        
    return [" ".join(a) for a in Taglist]
    