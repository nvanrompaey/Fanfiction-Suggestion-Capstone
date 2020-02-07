import numpy as np
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import Text
from nltk.probability import FreqDist


class StyleFinder():
    
    
    def __init__(self,fic):
        
        self.fic = fic
        self.words = word_tokenize(self.fic) #Tokenize up the words
        self.sentences = sent_tokenize(self.fic) #Tokenize up with sentences
        self.text = Text(self.words) # Easier to work around getting unique words
        self.fdist = FreqDist(self.text) # Get the frequency of a word, used for certain frequencies
        self.sent_len = [len(sentence.split()) for sentence in self.sentences] # Get the length of each sentence
        self.word_len = [len(word) for word in set(self.words)]
        
    def sentencel_mean(self):
        #Mean sentence length
        return np.mean(self.sent_len)
    
    def sentencel_std(self):
        #Standard Deviation sentence length
        return np.std(self.sent_len)
    
    def lex_diversity(self,size):
        #Take the number of unique tokens per 'size' number of words and divide it by the number of words total.
        return (len(set(self.text))*size)/len(self.text)
 
    def wordl_mean(self):
        #Mean word length
        return np.mean(self.word_len)
    
    def wordl_std(self):
        # Standard Deviation word length
        return np.std(self.word_len)
    
    def token_frequency(self,wordy,size):
        # Find the frequency of a given word (or punctuation) per 'size' number of words
        # FreqDist is an iterable, and .N() returns all the unique items 
        return (self.fdist[wordy]*size)/self.fdist.N()
        
        
        
    
    