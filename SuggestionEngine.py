import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def lazyPOSparser(stringinq):
    #Takes a string as a name you want the new row to be called, 
    # and a dataframe + column on which to perform the transformation
    tokenized = sent_tokenize(stringinq)
    Taglist = []
    for i in tokenized:
        # DO THIS: minimize words and add dummy column
        wordsList = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(wordsList)
        Taglist.append([i[1] for i in tagged])
    flat_list = []
    for sublist in Taglist:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def bad_analyzer(n):
    return None
    
    
    
def find_best_match(df,POSseries,n1,n2,story):
    # Takes a series based on a matrix, an ngram value pair, and a story index to compare.
    # Spits out a recommended story title and its POS shape
    tfidf = TfidfVectorizer(ngram_range=(n1,n2))
    POSjoined = POSseries.apply(lambda x: " ".join(x).replace('\'',''))
    POSTFIDF = tfidf.fit_transform(POSjoined)
    feature_freq_df = pd.DataFrame.sparse.from_spmatrix(POSTFIDF)
    recc = np.argmax([cosine_similarity(feature_freq_df.iloc[story].to_numpy().reshape(-1,1),
                                 feature_freq_df.iloc[x].to_numpy().reshape(-1,1)) for x in range(1000)])
    return df['title'].iloc(axis=0)[recc],df['story_shape'].iloc[recc]
    
    
    