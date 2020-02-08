# Content-Based Style-Oriented Fiction Suggestion

## Introduction to Fiction Searching

Often times, we're introduced to a piece of fiction by hearsay, genre, author, or rating. We might want a 5-star Mystery to read on a flight, or a 3/10 Romance to stop the boredom of a noisy busride. But among those genres, there's styles. An author might use different styles, or multiple authors might use a single style, but we wouldn't know that from the genre and authorship alone, and especially not by rating. Hearing about a style from one's friends is fairly unreliable, to boot.

This capstone seeks to answer the question: Can fiction be recommended based on writing style?

For the purposes of this capstone, I've taken an unnoficial collection of metadata and the stories themselves from the My Little Pony: Friendship is Magic fanfiction site FimFiction, and attempted to measure differences in writing style, before creating a recommendation engine.

For the purposes of this project, I was only concerned with writing style, and wound up ignoring everything else, including ignoring author, content tag, genre tags, character tags,rating, date, existence of prequel or coverimage, and author followers.

## Data

The data itself was organized as a .json with links to a series of epub files in many separate folders, organized by first letter, then author name, then fiction.
The columns I implimented were as follows:

| Indecies | Column | Description                       |
|----------|:------:|----------------------------------:|
| Story ID |archive| a path to the epub for the story |
| | title | the title of the story|
| | date_published| the date the story was published. This is sometimes blank|
| | date_updated| the date of the most recent update for the story|
| | description_html| The long description in HTML format|
| | num_chapters| how many chapters are in the story|
| | num_views| how many views the story has, not including individual chapter views|
| | num_dislikes| how many dislikes the story has|
| | num_likes | how many likes the story has|
| | num_words | the total number of words across chapters|
| | rating | the likes:dislikes ratio, rounded|
| | short_description| a shorter description, usually a sentence or two|
| | has_prequel | a simple dummy column for whether the story has a prequel or not|
| | has_cover_image| a simple dummy column for whether the story has a cover image or not|
| | completion_status| Complete, Incomplete, On Hiatus, Cancelled, later became dummy columns|
| | content_rating| Everyone, Teen, Mature, later became dummy columns|
| | author_name| Author of the story, by user. Does not necessarily mean the one who wrote the story|
| | author_num_followers| The number of followers the author has|
| | total_num_views| The total number of views, including separate chapter views|
| | tag_names| a list of tags for each story, not including series tags. Later became dummy columns|

Another table was added at the end of the first day, which was organized in the following way:

| Indecies | Column | Description                       |
|----------|:------:|----------------------------------:|
|Date| Daily Users| Number of users that joined on that day|
| | Total Users| Total number of users as of that day|

The Total Users was joined on the date_published, using date_updated if no date_published was available.

There's a lot possible with so much data, and although an initial plan promised a predictive algorithm to determine how well-rated a story might be based on a variety of these factors, this project ultimately went the direction of writing style recommendation. 

**The final table with which I worked became:**

| Indecies | Column | Description                       |
|----------|:------:|----------------------------------:|
| Story ID| title| the title of the story|
| | archive| the filepath to the story's epub file|
| | description_html| the long description in HTML form|
| | descrption_long| the description stripped of HTML|
| | word_count| the total number of words in the story|
| | story_text| the actual text of the story, pulled out from the epub file|


## Methods

**A quick note on Data Cleaning**

A great deal of data cleaning was done during the first day of work, but the data cleaning ultimately became  as simple as pulling out the epub files using the following code, with beautiful soup and an epub_conversion plugin:

```python
def textfinder(name):
    # 'name' is the name of the file in question being converted
    # Spits out a text file without html markings
    book = open_book(f"FData/{name}")
    text = CETL(book)
    soup = BeautifulSoup(" ".join(text))
    return soup.get_text()

def fic_col_monster(df,col,newcol,n):
    df2 = df.iloc[:n]
    #Takes a dataframe, a column in question, and spits out a 
    # new column with the name newcol that prints super cool stuff
    df2[newcol] = df2[col].apply(lambda x: textfinder(x))
    return df2
```

**First Model**
The first model made use of a TF-IDF vectorizor, by Scikit-Learn, but because this capstone was interested in writing style, the words themselves were ignored. Instead, each piece of story text was converted into a parts-of-speech tag by using ntlk's pos_tag() function (called 'POSparser'). The idea is to strip out the meaning of a sentence in order to only focus on the basic grammatical structures.

This method converted text like so:

- *Nopony could remember a time when it had rained so hard. The normally dry and dusty plains of Appleloosa and the surrounding area were flooded with the heavy rainfall.*

to

- NNP MD VB DT NN WRB PRP VBD VBN RB JJ . DT RB JJ CC JJ NNS IN NNP CC DT VBG NN VBD VBN IN DT JJ NN .


A second similar method converted text again by simplifying down all Nouns, Adjectives, Verbs, Adverbs, and Posessives, turning the sentence into something more like:

- NN MD VB JJ NN WRB NN VB VB RB JJ . JJ RB JJ CC JJ NN IN NN CC JJ VB NN VB VB IN JJ JJ NN .


A third, hierarchical method would have converted text yet further by making each phrase into a 'Noun Phrase' or 'Verb Phrase', which can be stacked on top of each other into consecutive tuples. Doing so would have redued the number of N-grams to worry about, likely 3, while also reducing the number of features the transformer would need to deal with, and the cosine similarity function.

However, with the first and second methods, the recommender was very, very slow, only capable of dealing with up to 4-5 grams using the TF-IDF before running far too long. However, it did give results, although those results were only somewhat useful, given the few n-grams they dealt with. Unfortunately, this highlighted a certain issue when my first significant result using the second method and 1000 stories was calculated.  


**Second Model**
The second model dropped the use of Parts of Speech entirely, and relied instead on punctuation, stop words, and word/sentence length and diversity. Not only is this a better metric, but it's a much faster calculation than a conversion to parts of speech then TF-IDF.

For this model, a custom count vectorizer was put together using the following functions:

|Function| Role|
|--------|-----|
|sentencel_mean |The Mean sentence length |
|sentencel_std | The Standard Deviation sentence length |
|wordl_mean| The Mean word length|
|wordl_std| The Standard Deviation word length|
|lex_diversity| The lexical diversity, that is, number of unique words per (size) words|
|token_frequency| The times a token appeared per (size) words|

The lexical diversity was found as follows:

```python
    def lex_diversity(self,size):
        #Take the number of unique tokens times 'size' number of words and divide it by the number of words.
        return (len(set(self.text))*size)/len(self.text)
```

The token frequency was found as follows:

```python
    def token_frequency(self,wordy,size):
        # Find the frequency of a given word (or punctuation) per 'size' number of words
        # FreqDist is an iterable, and .N() returns all the unique items 
        return (self.fdist[wordy]*size)/self.fdist.N()
```

token frequency used an odd nltk iterable known as Fdist, which gives a count for every token, as well as a N() function, which gives the total combined count of all tokens, including punctuation.

The token frequency function was used in order to pull out frequencies for specific stop words and punctuation. Some authors use a lot of commas, some use many quotes, and others use lots of em-dashes.

Likewise, some authors use lots of 'and', and 'that' and 'which', or 'as' and 'ing' and 'maybe', and such.

Considered tokens were swear words and more 'inappropriate' language, but each story already has tags for such content, and it was not necessary. Paragraph mean and standard deviation length was considered too, but the epub reader didn't account for paragraphs as well as was necessary.

The full list of used punctuation tokens is:
- , : ; ... ! - -- ) " '

The full list of stop word tokens is:
- and but however said while as ing though with cause to this that why how if then there might maybe its/it's

**Cosine Similarity**

In order to find the best


### Clustering
Afterwards, a K-means clustering algorithm from Scikit-Learn was used on the resulting Style Array, in order to predict clusters, and furthermore, figure out a good number of clusters. Unfortunately, I only had 10,000 stories to work with, and only 10,000 of the earlier stories, due to time constraints. That said, clustering seems to have worked to a point. The strongest Silhouette score was for 2, but there were some small rises. But as it turns out, there's one very, very strong category, and then a fairly strong secondary category. The very, very strong category is, in fact, all the stories which have text that are simply 'Chapter 1' and similar. This is fairly disappointing.


The Silhouette scores from 2 (0, technically) to 100 attempted clusters are as follows:

      [0.        , 0.        , 0.80917277, 0.51369042, 0.45536644,
       0.44886409, 0.40053898, 0.3688201 , 0.34926752, 0.34056985,
       0.32540674, 0.33694906, 0.33538317, 0.30737511, 0.31022552,
       0.29180665, 0.29054801, 0.27223126, 0.27354553, 0.27211477,
       0.29422657, 0.27790627, 0.24832777, 0.23594082, 0.23871477,
       0.23595804, 0.24128727, 0.24318867, 0.24635007, 0.25594305,
       0.23350688, 0.23026246, 0.24046917, 0.24677411, 0.22909061,
       0.22802436, 0.23287203, 0.22074746, 0.23454793, 0.23365307,
       0.22784268, 0.23010578, 0.21680028, 0.22469945, 0.22239729,
       0.2257877 , 0.23138595, 0.22790048, 0.21658115, 0.21759887,
       0.22070464, 0.21086759, 0.21247999, 0.21278687, 0.21529324,
       0.20928312, 0.20693424, 0.21165831, 0.20762206, 0.21906116,
       0.21058042, 0.20662974, 0.21712709, 0.20243966, 0.21219696,
       0.20613616, 0.19830792, 0.200244  , 0.20168743, 0.20227685,
       0.19548117, 0.19039498, 0.20439136, 0.19704521, 0.19393076,
       0.19959182, 0.1995872 , 0.1900777 , 0.19544654, 0.18601431,
       0.19795794, 0.18977326, 0.18897673, 0.18888368, 0.1921345 ,
       0.18269053, 0.19146888, 0.18544418, 0.18530267, 0.19528654,
       0.18808529, 0.18415199, 0.1868209 , 0.1898838 , 0.18607265,
       0.18330936, 0.18881078, 0.18019617, 0.1836953 , 0.17760937]
       
       
The graphs are as follows:

![Silhouette Graph 2](https://github.com/nvanrompaey/Fanfiction-Suggestion-Capstone/tree/master/Img/Silhouette2.png)

![Silhouette Graph 4](https://github.com/nvanrompaey/Fanfiction-Suggestion-Capstone/tree/master/Img/Silhouette4.png)

![Silhouette Graph 10](https://github.com/nvanrompaey/Fanfiction-Suggestion-Capstone/tree/master/Img/Silhouette10.png)



## Conclusions

It's a bit odd, how the Clustering failed, but the Second Model succeeded on finding 'similar-style' fiction, to a very high degree. Most cosine similarity scores were 0.992-0.999 with a mean of 0.97. The cosine similarity actually picks up on the small differences fairly significantly.

Clearly there's more work to do. The featurization can be tuned in order to account for the most common differences, or even account for every single stop word and its usage.

But until then, I'll have to say 'I do not know whether fiction can be recommended based on writing style.' The differences in the approaches' successes are very large.


## Limitations

The reason for choosing Fanfiction in the first place is its difficulty to work with as a datasource. Often, fanfiction is grammatically incorrect to a point, and stripping out poorly-written grammar is difficult, or at least resource-intensive. In terms of Lexical Diversity, this means the misspellings of 'and': 'nda', 'adn', 'nad', and so on, mean there will be more unique words overall, and with enough misspelling density, the Lexical Diversity goes straight up without being 'earned', so to speak. Furthermore, it would prevent certain stop words from being recognized, although that part is a bit of an advantage--if a story is poorly written, it probably shouldn't be recommended in the first place. Fortunately, with the second method, but unfortunately not with the first, grammatical errors would not be counted as certain parts of speech that they otherwise would be.

The reason for choosing My Little Pony fanfiction was its interesting use of Naming. Most characters are named after improper nouns. Twilight Sparkle is made up of two reasonable words: twilight and sparkle. Rainbow Dash is also made up of two regular nouns: rainbow and dash. Any function that tries to distinguish the proper nouns would have a great deal of difficulty with these particular proper nouns, and the multitude of similar ones. The use of lots of accents in dialogue, such as Applejack's, eg. "Ah didn' think you'd care" and such writing styles, makes it hard to parse out certain words when in dialogue or first person narrative.

Lastly, the sheer size of the data was an issue. There were over 180,000 stories, with a total of 2.37 Billion words across them. That much data was a hassle to deal with even on the small scale of 100 stories, much less 10,000, much much less 180,000. This capstone was unable to get its AWS working within a reasonable length of time, and therefore this project could not continue beyond 10,000. These conclusions assume the first 10,000 stories. Notably, there was a period of time after which the minimum story length became 1,000.

## Sources

Scikit-Learn (TFIDF functions, Cosine Similarity, K-means functions)
NLTK (NLP functions, POS_tag function)
FIMFiction unofficial story archive (and metadata)
FIMFiction official Statistics page (in csv form)
"Stylometry-based Approach for Detecting Writing Style Changes in Literary Texts" for Inspiration in style approaches



