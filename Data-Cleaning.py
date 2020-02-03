import numpy as np
import pandas as pd

def safeget(x,s,a,b): 
    #If a dictionary exists within dictionary X, 
    # then try to find the value of key 'b' within the secondary
    if x[s].get(a) is not None:
        return x[s].get(a).get(b)
    else:
        return None

    
def emptynans(df,columns):
    #Takes Nan values for a column and fills with empty strings.
    for i in columns:
        df[columns].fillna("",inplace=True)

def dummynans(df,columns):
    #Takes columns where nans are important and creates a 'has' where Nan is 0 and anything else is 1.
    for i in columns:
        df[f"has_{i}"] = ficdf[i].apply(lambda x: x is not None).astype(int)
    
    
    
    
    
    
def FicData(link):
    #link is 'FData/index.json'
    testdf = pd.read_json(link)
    testdf = testdf.transpose()
    testdf.set_index('id',inplace=True)
    
    #Dropping unneeded and redundant or single-value Columns
    # color is unneeded. Status, Published, and Submitted are all True for all values
    # Date modified is only present in a fraction of fics. URL contains redundant information. Chapters is too indepth.
    ficdf = testdf.drop(['color','status','published','submitted','date_modified','url','chapters'],axis=1)
    #Dropping rows where the story has 0 words.
    ficdf = ficdf.drop(ficdf.index[ficdf['num_words'] == 0],axis=0)
    
    #Fixing Nans
    # In this case, published dates
    ficdf['date_published'].fillna(ficdf['date_updated'],inplace=True)
    # Instead of dropping rows or keeping nans, empty strings will flow well with grammar-tracking
    emptynans(ficdf,['description_html','short_description','title'])
    
    #Creating dummy columns based on Nans
    dummynans(ficdf,['prequel','cover_image'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    