from flask import Flask,render_template,request,jsonify
import numpy as np
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from StyleFinder import StyleFinder as SF
app = Flask(__name__)


@app.route('/',methods=['GET'])
def index():
    return render_template('fictionsuggestion.html')


@app.route('/post',methods=['POST'])
def get_pos():
    user_data = request.json
    sentence = user_data['sentence']
    posted = _poster(sentence)
    return jsonify({"posted":posted})

def _poster(sent):
    tokenized = word_tokenize(sent)
    tagged = nltk.pos_tag(tokenized)
    tagstring = " ".join([a[1] for a in tagged])
    return tagstring

@app.route('/reverse',methods=['POST'])
def get_rev():
    user_data = request.json
    para=user_data['paragraph']
    smean,wmean,ldiv,maxtoken=_reverser(para)
    
    return jsonify({"smean":smean,"wmean":wmean,"ldiv":ldiv,"maxtoken":maxtoken})

def _reverser(sent):
    stoplist = ["and","but","however","said","while","as","ing",
                "though","with","cause","to","this","that","why",
                "how","if","then","there","might","maybe","its"]
    v = np.zeros(len(stoplist))
    style = SF(sent)
    smean = style.sentencel_mean()
    wmean = style.wordl_mean()
    ldiv = style.lex_diversity(1000)
    for a in range(len(stoplist)):
        v[a] = style.token_frequency(stoplist[a],1000)
    maxtoken = stoplist[np.argmax(v)]
    
    return smean,wmean,ldiv,maxtoken
    
    
    







if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=False)