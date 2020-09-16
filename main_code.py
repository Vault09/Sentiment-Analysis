import pandas as pd #pandas is an easy to use open source data analysis and manipulation tool
import gzip    #This module provides a simple interface to compress and decompress files 


false = False   #assign <To debug the error :name 'false' is not defined >
true = True   #assign <To debug the error :name 'true' is not defined >
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
 
def getDF(path): 
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
mag_data = getDF('Magazine_Subscriptions.json.gz')
mag_data.head(5)
print(mag_data.head())
mag_data['reviewText'].fillna('', inplace = True)
mag_data=mag_data.drop(['vote' , 'image' , 'style','verified'],axis=1) #  Drop these columns as they do not assist in text classification
mag_data = mag_data[['asin', 'summary', 'reviewText', 'overall', 'reviewerID', 'reviewerName', 'reviewTime',
      'unixReviewTime']]# Rearrange the columns in dataframe in above order
print(mag_data.head())
mag_data = mag_data.head(20000)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import string
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer


mag_data['reviewText'] = mag_data['reviewText'].str.lower() #To convert the text "reviewText in 
from textblob import TextBlob

def detect_polarity(text):
    return TextBlob(text).sentiment.polarity
mag_data['polarity'] = mag_data['reviewText'].apply(detect_polarity)
print(mag_data['polarity'].head())




#Distribution of Polarity


num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(mag_data.polarity, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of polarity')
plt.show();




#Tokenize sentences in the "reviewText" into phrases
#from nltk.tokenize import sent_tokenize
mag_data['tokenized_sents'] = mag_data.apply(lambda row: nltk.sent_tokenize(row['reviewText']), axis=1)
mag_data['tokenized_sents'].head()





#from nltk.tokenize import word_tokenize
#mag_data['tokenized_word'] = mag_data.apply(lambda row: nltk.word_tokenize(row['reviewText']), axis=1)
#mag_data['tokenized_word'].head(2)




#Tokenize sentences in the "reviewText" into words



def identify_tokens(row):
    review = row['reviewText']
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

mag_data['tokenized_word'] = mag_data.apply(identify_tokens, axis=1)
mag_data['tokenized_word'].head()



from nltk.corpus import stopwords
stop_words_list = set(stopwords.words("english"))
mag_data['stop_words'] = mag_data.apply(lambda row: set(stopwords.words("english")), axis=1)
mag_data['stop_words'].head(2)
# Stop Words 


#remove stopwords
mag_data['filt_words'] = mag_data['tokenized_word'].apply(lambda x: [item for item in x if item not in stop_words_list])


#tokenized words after stopwords removed
mag_data['filt_words'].head()





#The filtered words are further brought to their root /stem word
#from nltk.stem import PorterStemmer
def Stemming_Words(Words):
    Ps = PorterStemmer()
    Stemmed_Words = []
    for m in Words:
        Stemmed_Words.append(Ps.stem(m))
    return Stemmed_Words
mag_data['stem_words'] = mag_data.apply(lambda row: Stemming_Words(row['filt_words']), axis=1)
mag_data['stem_words'].head()




#Stemmed words are further lemmatized
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
def Lemmatizing_Words(Words):
    Lm = WordNetLemmatizer()
    Lemmatized_Words = []
    for m in Words:
        Lemmatized_Words.append(Lm.lemmatize(m))
    return Lemmatized_Words

mag_data['Lemmatized_Words'] = mag_data.apply(lambda row: Lemmatizing_Words(row['stem_words']), axis=1)
print(mag_data['Lemmatized_Words'].head())



#Create the dictionary of the preprocessed words using gensim library


from gensim import corpora
# Build the dictionary
mydict = corpora.Dictionary(mag_data['stem_words'])
print("Total unique words:")
print(len(mydict.token2id))
print("\nSample data from dictionary:")
i = 0
# Print top 4 (word, id) tuples
for key in mydict.token2id.keys():
    print("Word: {} - ID: {} ".format(key, mydict.token2id[key]))
    if i == 3:
        break
    i += 1
    
    
#number of reviews with overall rating wise.
print(mag_data.overall.value_counts())


#plotting the reviews distribution by overall rating
import matplotlib.pyplot as plt
Sentiment_count=mag_data.groupby('overall').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['reviewText'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

import numpy as np

#mapping sentiment of each review on behalf of their overall score

import matplotlib.pyplot as plt 

print("Number of rows per star rating:")
print(mag_data['overall'].value_counts())

# Function to map stars to sentiment
def map_sentiment(stars_received):
    if stars_received <= 2.0:
        return -1
    elif stars_received == 3.0:
        return 0
    else:
        return 1
# Mapping stars to sentiment into three categories
mag_data['sentiment'] = [ map_sentiment(x) for x in mag_data['overall']]
# Plotting the sentiment distribution
plt.figure()
pd.value_counts(mag_data['sentiment']).plot.bar(title="Sentiment distribution in DataFrame")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in df")
plt.show()



colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
explode = (0.1, 0, 0)

pd.value_counts(mag_data['sentiment']).plot.pie(title="Sentiment distribution in DataFrame" ,explode=explode, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()





# Function to retrieve top few number of each category so that our model is not biased to one particular class of sentiment
def get_top_data(top_n =200):
    top_data_df_positive = mag_data[mag_data['sentiment'] == 1].head(top_n)
    top_data_df_negative = mag_data[mag_data['sentiment'] == -1].head(top_n)
    top_data_df_neutral = mag_data[mag_data['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small

# Function call to get the top  from each sentiment
top_data_df_small = get_top_data(top_n =200)
                                 

# After selecting top few samples of each sentiment
print("After segregating and taking equal number of rows for each sentiment:")
print(top_data_df_small['sentiment'].value_counts())
print(top_data_df_small.head())



#Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X = top_data_df_small['Lemmatized_Words']
y = top_data_df_small['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)



def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
    
    print ('Creating bag of words...')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    
    # In this example features may be single words or two consecutive words
    # (as shown by ngram_range = 1,2)
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc,ngram_range = (1,3) , lowercase=False)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)
    print(train_data_features)
    
    # Convert to a NumPy array for easy of handling
    print("array of train_data_features")
    train_data_features = train_data_features.toarray()
    print(train_data_features)
    print("displayed")
    
    # tfidf transform
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()
    print("tfidf_features")
 #   print(tfidf_features)

    # Get words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print('vocab')
            # print(vocab)
   
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf
    

vectorizer, vocab, train_data_features, tfidf_features, tfidf  = \
    create_bag_of_words(X_train)







#create new dataframe bow_dictionary to store phrase/word , its count and tfidfvalue

bow_dictionary = pd.DataFrame()
bow_dictionary['ngram'] = vocab
bow_dictionary['count'] = train_data_features[0]
bow_dictionary['tfidf_features'] = tfidf_features[0]

# Sort by raw count
bow_dictionary.sort_values(by=['count'], ascending=False, inplace=True)
# Show top 10
print(bow_dictionary.head(10))








def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 100,random_state = 0)
    ml_model.fit(features, label)
    print ('Finished')
    return ml_model




#convert the test data same as we did on the train data 
test_data_features = vectorizer.transform(X_test)
# Convert to numpy array
test_data_features = test_data_features.toarray()



test_data_tfidf_features = tfidf.fit_transform(test_data_features)
print(test_data_tfidf_features)
# Convert to numpy array
test_data_tfidf_features = test_data_tfidf_features.toarray()



print(test_data_tfidf_features)


#training the model
ml_model = train_logistic_regression(tfidf_features, y_train)



#checking the accuracyof our model
predicted_y = ml_model.predict(test_data_tfidf_features)
print("Real Sentiment")
print(y_test)

print("Predicted Sentiment by our model")
print(predicted_y)
correctly_identified_y = predicted_y == y_test
accuracy = np.mean(correctly_identified_y) * 100
print ('Accuracy = %.0f%%' %accuracy)


import sqlite3


from flask import Flask, render_template, request,redirect, url_for

app = Flask(__name__,template_folder="templates", static_folder="static")


@app.route('/')
def index():
    return render_template('indexsa.html')

rev=["object"]
@app.route('/home')
def ren_gv():
    return render_template('homesa.html')
@app.route('/home', methods=['POST'])
def getValue():
    global rev
    global res
    global product
    if request.form.get("product1"):
        product=1
    elif request.form.get("product2"):
        product=2
    elif request.form.get("product3"):
        product=3
    res = request.form['message']
    rev=[res]
    print(res)
    print(product)
    
    return redirect(url_for('output'))

 
    




@app.route('/output')
def output():
    global predicted_sentiment
    global instance_data_tfidf_features
    global senti

    instance_data_features = vectorizer.transform(rev)
# Convert to numpy array
    instance_data_features = instance_data_features.toarray()



    instance_data_tfidf_features = tfidf.fit_transform(instance_data_features)
    print(test_data_tfidf_features)
# Convert to numpy array
    instance_data_tfidf_features = instance_data_tfidf_features.toarray()





    predicted_sentiment = ml_model.predict(instance_data_tfidf_features)
    if ml_model.predict(vectorizer.transform(rev)) == -1:
        senti = "negative"
    elif ml_model.predict(vectorizer.transform(rev)) == 0:
        senti = "neutral"
    elif ml_model.predict(vectorizer.transform(rev)) == 1:
        senti = "positive"
    print(senti)
    print(predicted_sentiment)
    sa=int(predicted_sentiment)
    if sa == -1:
        s = "negative"
    elif sa == 0:
        s = "neutral"
    elif sa == 1:
        s = "positive"
    print(s)
    return render_template("outputsa.html", s1=s)

@app.route('/stat' , methods = ['POST','GET'])
def stat():
    ps=int(predicted_sentiment)
    conn=sqlite3.connect('pythonDB3.db')
    c=conn.cursor()
    def create_table():
        c.execute('CREATE TABLE IF NOT EXISTS RecordONE(Review TEXT, Sentiment_Value INTEGER, Product INTEGER)')
    def data_entry():
        c.execute("INSERT INTO RecordONE (Review, Sentiment_Value, Product) VALUES(?, ?, ?)", (res, ps, product))
        conn.commit()
    
    create_table()
    data_entry()

    c.close()
    conn.close()

    print("Probability of Predicted Sentiment")
    prob_of_predicted_sentiment =  ml_model.predict_proba(instance_data_tfidf_features)
    print(prob_of_predicted_sentiment)
    prob_of_predicted_sentiment.shape



    list = [prob_of_predicted_sentiment[0,0] ,prob_of_predicted_sentiment[0,1] ,prob_of_predicted_sentiment[0,2]]
    print(list)


    import os
    import time

    labels = 'Negative', 'Neutral', 'Positive'

    explode = (0, 0, 0.1) 
    
    fig1, ax1 = plt.subplots()
    ax1.pie(list ,explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    ngn="img1" + str(time.time()) + ".png"
    

    
    for filename in os.listdir('static/'):
        if filename.startswith('img1_'):
            os.remove('static/' + filename)
    plt.show()
    fig1.savefig("static/" + ngn ,bbox_inches="tight")

    dat=sqlite3.connect('pythonDB3.db')
    query=dat.execute("SELECT * FROM RecordONE")
    cols=[column[0] for column in query.description]
    results=pd.DataFrame.from_records(data=query.fetchall(), columns = cols)

    print(results)
    d1=results[results['Product']==1]
    d2=results[results['Product']==2]
    d3=results[results['Product']==3]    
    print(d1)
    print(d2)
    print(d3)
    v1=d1['Sentiment_Value']
    v2=d2['Sentiment_Value']
    v3=d3['Sentiment_Value']


    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
#explode = (0.1, 0, 0)

    dtf=pd.value_counts(v1).plot.pie(title="Sentiment distribution in DataFrame" , colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
#dtf.plot()
    ngn1="mtor1" + str(time.time()) + ".png"
    

    
    for filename in os.listdir('static/'):
        if filename.startswith('mtor1_'):
            os.remove('static/' + filename)
    p=plt.show()
    dtf.figure.savefig("static/" + ngn1, bbox_inches="tight")



    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
#explode = (0.1, 0, 0)

    dtf2=pd.value_counts(v2).plot.pie(title="Sentiment distribution in DataFrame" ,  colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
#dtf.plot()
    ngn2="mtor2" + str(time.time()) + ".png"
    

    
    for filename in os.listdir('static/'):
        if filename.startswith('mtor2_'):
            os.remove('static/' + filename)
    p=plt.show()
    dtf2.figure.savefig("static/" + ngn2, bbox_inches="tight")


    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
#explode = (0.1, 0, 0)

    dtf3=pd.value_counts(v3).plot.pie(title="Sentiment distribution in DataFrame" ,  colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
#dtf.plot()
    ngn3 ="mtor3" + str(time.time()) + ".png"
    

    
    for filename in os.listdir('static/'):
        if filename.startswith('mtor3_'):
            os.remove('static/' + filename)
    plt.show()
    dtf3.figure.savefig("static/" + ngn3, bbox_inches="tight")
  
    return render_template("statsa.html",img1 = ngn, mtor1 = ngn1, mtor2 = ngn2, mtor3 = ngn3)

@app.route('/about')
def aboutt():
    return render_template("aboutsa.html")

@app.route('/sett')
def sett():
    return render_template("settingssa.html")

@app.route('/contacts')
def contacts():
    return render_template("contactsa.html")

if __name__=="__main__":
    app.run(host="localhost", port=int("777"))
#rev = ["bad product"]
#print(ml_model.predict(vectorizer.transform(rev)))
#if ml_model.predict(vectorizer.transform(rev)) == -1:
    #senti = "negative"
#elif ml_model.predict(vectorizer.transform(rev)) == 0:
    #senti = "neutral"
#elif ml_model.predict(vectorizer.transform(rev)) == 1:
    #senti = "positive"
#from flask import Flask, render_template
#app= Flask (__name__,template_folder="templates")
#s=senti
#print(s)
#@app.route('/')
#def output():
    #return render_template("output.html", s1=s)
#if __name__=="__main__":
    #app.run(host="localhost", port=int("778"))
    
    


  
        

