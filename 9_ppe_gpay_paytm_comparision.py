import nltk
import pandas as pd
import numpy as np
import os
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS 
from nltk.sentiment.vader import SentimentIntensityAnalyzer


%matplotlib

os.chdir('F:\Board infinity\phonepe')


######  phonepe data analysis

df_phonepe1 = pd.read_csv('phonepe_reviews.csv')
df_phonepe1[df_phonepe1.Source=='Twitter'].index

df_phonepe=df_phonepe1.drop(index=df_phonepe1[df_phonepe1.Source=='Twitter'].index)


def sub_processing(text):
    text = re.sub(r'[^ A-Z a-z .\',]+','',text)
    text = re.sub(r'[.]+','.',text)
    text = re.sub(r'\s+',' ',text)
    return text


df_phonepe_processed = df_phonepe.copy()
df_phonepe_processed['Reviews'] = df_phonepe['Reviews'].apply(sub_processing)


df_phonepe_processed.head()

nlp = spacy.load('en_core_web_lg')

def word_cloud(df): 
    comment_words = '' 
    stopwords = spacy.lang.en.stop_words.STOP_WORDS


    for val in df: 
        
        val = str(val) 
        tokens = val.split() 
      
        # Converts each token into lowercase 
        for i in range(len(tokens)):  
            tokens[i] = tokens[i].lower() 
      
        comment_words += " ".join(tokens)+" "
  
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 

    return wordcloud




################################################################################################
#########  comparision between phonepe, googlepay, paytm
'''
(1) we will check if google pay and paytm are mentioned in the reviews of phonepe
  if yes, then we will find are the positive or negative reviews. 
  also, a.  phonepe positive review = google pay negative review
        b.  phonepe nagative review = google pay positive review
        
        same goes for paytm Reviews.
  
(2) we will also check if phonepe reviews are present in googlepay and paytm reviews
  if yes, then we will find are the positive or negative reviews. 
'''

############ (1)  a. google pay in phonepe reviews

df_google_pay = pd.DataFrame()
df_google_pay['Reviews'] = df_phonepe_processed['Reviews'].apply(lambda x: x.lower() if ('google' in x.lower())  else np.nan)
df_google_pay.dropna(inplace=True)

top_words_wc_temp = word_cloud(df_google_pay.Reviews).words_

## reviews related to vannurappa are advertisement so we are droping all orws which are having varrunappa mentioned in it.
df_phonepe_processed['Reviews'] = df_phonepe_processed['Reviews'].apply(lambda x: x if ('vannurappa' not in x.lower())  else np.nan)
df_phonepe_processed.dropna(inplace=True)

## again creating google pay dataframe
df_google_pay = pd.DataFrame()
df_google_pay['Reviews'] = df_phonepe_processed['Reviews'].apply(lambda x: x.lower() if ('google' in x.lower())  else np.nan)
df_google_pay.dropna(inplace=True)

top_words_wc_temp = word_cloud(df_google_pay.Reviews).words_


########   analysing key words which can influence the sentiment
def pos_extractor(review):
    
    for  token in nlp(review):
        if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN':
            nouns.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'VERB':
            verbs.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADJ':
            adjs.append(token.text)

nouns = []
adjs = []
verbs = []

df_google_pay.Reviews.apply(pos_extractor)

 
df_gpay_nouns = pd.DataFrame({'nouns':nouns})
df_gpay_nouns = df_gpay_nouns.nouns.apply(lambda x: x.lower())
df_gpay_nouns = pd.DataFrame(df_gpay_nouns)
top_100_nouns = df_gpay_nouns.nouns.value_counts()[0:100]


df_gpay_adjs = pd.DataFrame({'adjs':adjs})
df_gpay_adjs = df_gpay_adjs.adjs.apply(lambda x: x.lower())
df_gpay_adjs = pd.DataFrame(df_gpay_adjs)
top_100_adjs = df_gpay_adjs.adjs.value_counts()[0:100]


df_gpay_verbs = pd.DataFrame({'verbs':verbs})
df_gpay_verbs = df_gpay_verbs.verbs.apply(lambda x: x.lower())
df_gpay_verbs = pd.DataFrame(df_gpay_verbs)
top_100_verbs = df_gpay_verbs.verbs.value_counts()[0:100]



## code to check the keywords which can enfluence the decision of the sentiment
## df_check is the temporary dataset to analyse the influenceing keyword and misclassification chances

df_check = pd.DataFrame()
df_check['Reviews'] = df_google_pay['Reviews'].apply(lambda x: x.lower() if ('unable' in x.lower())  else np.nan)
df_check.dropna(inplace=True) 

analyzer = SentimentIntensityAnalyzer()
df_check['scores'] = df_check.Reviews.apply(analyzer.polarity_scores)
df_check['compound_score_label'] = df_check.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )




## we are going to use vader for classifing positive and negative reviews and sentiment analysis


## vader analyser
analyzer = SentimentIntensityAnalyzer()
df_google_pay['scores'] = df_google_pay.Reviews.apply(analyzer.polarity_scores)
df_google_pay['compound_score_label'] = df_google_pay.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )
df_google_pay.compound_score_label.value_counts()

'''
pos        704
neg        377
neutral     41
'''
## word 'better than phone' in reviews refers to google pay and paytm not phone pe.
## though it is a positive word, the review still can be negative in nature
## we removing 'better than phone' from all selected reviews.

def keyword_analyzer(x):
    if ('worst' in x) or ('pathetic' in x) or ('unable' in x) or ('slower than other' in x) or ('better than phone' in x) or ('best than phone' in x) or ('not better than google' in x) or ('waste than google' in x):
        return 'neg'
    elif ('not better than phone' in x) or ('better than google' in x) or ('best than google' in x) or ('good than google' in x) or ('faster than google' in x) or ('better than other' in x):
        return 'pos'
    

df_google_pay['temp'] = df_google_pay['Reviews'].apply(keyword_analyzer)


for index in df_google_pay.index:
    if df_google_pay['temp'][index] != None:
        df_google_pay.compound_score_label[index]=df_google_pay['temp'][index]
        

df_google_pay.compound_score_label.value_counts()

ax=ax = sns.countplot(x='compound_score_label', data=df_google_pay)
plt.title('Analysis of Gpay comment in Phonepe reviews')
'''
pos        665
neg        417
neutral     40

there were total 665 reviews of phonepe in which response for phonepe is positive against google pay 
there were total 417 reviews of phonepe in which response for phonepe is negative against google pay

there are total 40 neutral reviews, most of them are advertigements
'''




############ (1)  b. paytm in phonepe reviews


df_paytm = pd.DataFrame()
df_paytm['Reviews'] = df_phonepe_processed['Reviews'].apply(lambda x: x.lower() if ('paytm' in x.lower())  else np.nan)
df_paytm.dropna(inplace=True)

top_words_wc_temp = word_cloud(df_paytm.Reviews).words_



########   analysing key words which can influence the sentiment
def pos_extractor(review):
    
    for  token in nlp(review):
        if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN':
            nouns.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'VERB':
            verbs.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADJ':
            adjs.append(token.text)

nouns = []
adjs = []
verbs = []

df_paytm.Reviews.apply(pos_extractor)

 
df_paytm_nouns = pd.DataFrame({'nouns':nouns})
df_paytm_nouns = df_paytm_nouns.nouns.apply(lambda x: x.lower())
df_paytm_nouns = pd.DataFrame(df_paytm_nouns)
top_100_nouns = df_paytm_nouns.nouns.value_counts()[0:100]


df_paytm_adjs = pd.DataFrame({'adjs':adjs})
df_paytm_adjs = df_paytm_adjs.adjs.apply(lambda x: x.lower())
df_paytm_adjs = pd.DataFrame(df_paytm_adjs)
top_100_adjs = df_paytm_adjs.adjs.value_counts()[0:100]


df_paytm_verbs = pd.DataFrame({'verbs':verbs})
df_paytm_verbs = df_paytm_verbs.verbs.apply(lambda x: x.lower())
df_paytm_verbs = pd.DataFrame(df_paytm_verbs)
top_100_verbs = df_paytm_verbs.verbs.value_counts()[0:100]



## code to check the keywords which can enfluence the decision of the sentiment
## df_check is the temporary dataset to analyse the influenceing keyword and misclassification chances

df_check = pd.DataFrame()
df_check['Reviews'] = df_paytm['Reviews'].apply(lambda x: x.lower() if ('better than other' in x.lower())  else np.nan)
df_check.dropna(inplace=True) 

analyzer = SentimentIntensityAnalyzer()
df_check['scores'] = df_check.Reviews.apply(analyzer.polarity_scores)
df_check['compound_score_label'] = df_check.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )




## we are going to use vader for classifing positive and negative reviews and sentiment analysis


## vader analyser
analyzer = SentimentIntensityAnalyzer()
df_paytm['scores'] = df_paytm.Reviews.apply(analyzer.polarity_scores)
df_paytm['compound_score_label'] = df_paytm.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )
df_paytm.compound_score_label.value_counts()

'''
pos        445
neg        199
neutral     85
'''
## word 'better than phone' in reviews refers to google pay and paytm not phone pe.
## though it is a positive word, the review still can be negative in nature
## we removing 'better than phone' from all selected reviews.

def keyword_analyzer(x):
    if ('worst' in x) or ('pathetic' in x) or ('unable' in x) or ('slower than other' in x) or ('better than phone' in x) or ('best than phone' in x) or ('not better than paytm' in x) or ('waste than patm' in x):
        return 'neg'
    elif ('not better than phone' in x) or ('better than paytm' in x) or ('best than paytm' in x) or ('good than paytm' in x) or ('faster than paytm' in x) or ('better than other' in x):
        return 'pos'
    

df_paytm['temp'] = df_paytm['Reviews'].apply(keyword_analyzer)


for index in df_paytm.index:
    if df_paytm['temp'][index] != None:
        df_paytm.compound_score_label[index]=df_paytm['temp'][index]
        
df_paytm.drop(columns=['temp'],inplace=True)
df_paytm.compound_score_label.value_counts()

ax = sns.countplot(x='compound_score_label', data=df_paytm)
plt.title('Analysis of paytm in phonepe reviews')

'''
pos        415
neg        230
neutral     84

there were total 415 reviews of phonepe in which response for phonepe is positive against paytm 
there were total 230 reviews of phonepe in which response for phonepe is negative against paytm

there are total 84 neutral reviews, most of them are advertigements
'''

########################################## 2  a. phonepe reviews in google pay dataset
'''
NOTE:
a.  google pay positive review = phonepe nagative review
b.  google pay negative review = phonepe positive review 
'''


df_google_pay1 = pd.read_csv('google_pay_reviews.csv')
df_google_pay1[df_google_pay1.Source=='Twitter'].index

df_google_pay=df_google_pay1.drop(index=df_google_pay1[df_google_pay1.Source=='Twitter'].index)

def sub_processing(text):
    text = re.sub(r'[^ A-Z a-z .\',]+','',text)
    text = re.sub(r'[.]+','.',text)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    return text


df_google_pay_processed = df_google_pay.copy()
df_google_pay_processed['Reviews'] = df_google_pay['Reviews'].apply(sub_processing)


df_google_pay_processed.head()


df_phonepe_in_gpay = pd.DataFrame()
df_phonepe_in_gpay['Reviews'] = df_google_pay_processed['Reviews'].apply(lambda x: x if ('phone' in x)  else np.nan)
df_phonepe_in_gpay.dropna(inplace=True)

## word cloud analysis
top_words_wc_temp = word_cloud(df_paytm.Reviews).words_


## pos analysis

def pos_extractor(review):
    
    for  token in nlp(review):
        if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN':
            nouns.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'VERB':
            verbs.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADJ':
            adjs.append(token.text)

nouns = []
adjs = []
verbs = []

df_phonepe_in_gpay.Reviews.apply(pos_extractor)

 
df_phonepe_in_gpay_nouns = pd.DataFrame({'nouns':nouns})
df_phonepe_in_gpay_nouns = df_phonepe_in_gpay_nouns.nouns.apply(lambda x: x.lower())
df_phonepe_in_gpay_nouns = pd.DataFrame(df_phonepe_in_gpay_nouns)
top_100_nouns = df_phonepe_in_gpay_nouns.nouns.value_counts()[0:100]


df_phonepe_in_gpay_adjs = pd.DataFrame({'adjs':adjs})
df_phonepe_in_gpay_adjs = df_phonepe_in_gpay_adjs.adjs.apply(lambda x: x.lower())
df_phonepe_in_gpay_adjs = pd.DataFrame(df_phonepe_in_gpay_adjs)
top_100_adjs = df_phonepe_in_gpay_adjs.adjs.value_counts()[0:100]


df_phonepe_in_gpay_verbs = pd.DataFrame({'verbs':verbs})
df_phonepe_in_gpay_verbs = df_phonepe_in_gpay_verbs.verbs.apply(lambda x: x.lower())
df_phonepe_in_gpay_verbs = pd.DataFrame(df_phonepe_in_gpay_verbs)
top_100_verbs = df_phonepe_in_gpay_verbs.verbs.value_counts()[0:100]



## code to check the keywords which can enfluence the decision of the sentiment
## df_check is the temporary dataset to analyse the influenceing keyword and misclassification chances

df_check = pd.DataFrame()
df_check['Reviews'] = df_phonepe_in_gpay['Reviews'].apply(lambda x: x.lower() if ('better than other' in x.lower())  else np.nan)
df_check.dropna(inplace=True) 

analyzer = SentimentIntensityAnalyzer()
df_check['scores'] = df_check.Reviews.apply(analyzer.polarity_scores)
df_check['compound_score_label'] = df_check.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )




################## we are going to use vader for classifing positive and negative reviews and sentiment analysis
## vader analyser
analyzer = SentimentIntensityAnalyzer()
df_phonepe_in_gpay['scores'] = df_phonepe_in_gpay.Reviews.apply(analyzer.polarity_scores)
df_phonepe_in_gpay['compound_score_label'] = df_phonepe_in_gpay.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )
df_phonepe_in_gpay.compound_score_label.value_counts()

'''
pos        445
neg        199
neutral     85
'''
## word 'better than phone' in reviews refers to google pay and paytm not phone pe.
## though it is a positive word, the review still can be negative in nature
## we removing 'better than phone' from all selected reviews.

def keyword_analyzer(x):
    if ('worst' in x) or ('pathetic' in x) or ('unable' in x) or ('slower than other' in x) or ('better than phone' in x) or ('best than phone' in x) or ('not better than paytm' in x) or ('waste than patm' in x):
        return 'neg'
    elif ('not better than phone' in x) or ('better than paytm' in x) or ('best than paytm' in x) or ('good than paytm' in x) or ('faster than paytm' in x) or ('better than other' in x):
        return 'pos'
    

df_phonepe_in_gpay['temp'] = df_phonepe_in_gpay['Reviews'].apply(keyword_analyzer)


for index in df_phonepe_in_gpay.index:
    if df_phonepe_in_gpay['temp'][index] != None:
        df_phonepe_in_gpay.compound_score_label[index]=df_phonepe_in_gpay['temp'][index]
        
df_phonepe_in_gpay.drop(columns=['temp'],inplace=True)
df_phonepe_in_gpay.compound_score_label.value_counts()
ax = sns.countplot(x='compound_score_label', data=df_phonepe_in_gpay)
plt.title('phonepe comment in gpay dataset')

'''
pos        415
neg        230
neutral     84

there were total 415 reviews of phonepe in which response for phonepe is positive against paytm 
there were total 230 reviews of phonepe in which response for phonepe is negative against paytm

there are total 84 neutral reviews, most of them are advertigements
'''



# 2.b phonepe reviews in paytm dataset

df_paytm1 = pd.read_csv('paytm_reviews.csv')
df_paytm1[df_paytm1.Source=='Twitter'].index

df_paytm=df_paytm1.drop(index=df_paytm1[df_paytm1.Source=='Twitter'].index)

def sub_processing(text):
    text = re.sub(r'[^ A-Z a-z .\',]+','',text)
    text = re.sub(r'[.]+','.',text)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    return text


df_paytm_processed = df_paytm.copy()
df_paytm_processed['Reviews'] = df_paytm['Reviews'].apply(sub_processing)


df_paytm_processed.head()


df_phonepe_in_paytm = pd.DataFrame()
df_phonepe_in_paytm['Reviews'] = df_paytm_processed['Reviews'].apply(lambda x: x if ('phone' in x)  else np.nan)
df_phonepe_in_paytm.dropna(inplace=True)

## word cloud analysis
top_words_wc_temp = word_cloud(df_paytm.Reviews).words_


## pos analysis

def pos_extractor(review):
    
    for  token in nlp(review):
        if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN':
            nouns.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'VERB':
            verbs.append(token.text)
        elif token.is_stop != True and token.is_punct !=True and token.pos_ == 'ADJ':
            adjs.append(token.text)

nouns = []
adjs = []
verbs = []

df_phonepe_in_gpay.Reviews.apply(pos_extractor)

 
df_phonepe_in_gpay_nouns = pd.DataFrame({'nouns':nouns})
df_phonepe_in_gpay_nouns = df_phonepe_in_gpay_nouns.nouns.apply(lambda x: x.lower())
df_phonepe_in_gpay_nouns = pd.DataFrame(df_phonepe_in_gpay_nouns)
top_100_nouns = df_phonepe_in_gpay_nouns.nouns.value_counts()[0:100]


df_phonepe_in_gpay_adjs = pd.DataFrame({'adjs':adjs})
df_phonepe_in_gpay_adjs = df_phonepe_in_gpay_adjs.adjs.apply(lambda x: x.lower())
df_phonepe_in_gpay_adjs = pd.DataFrame(df_phonepe_in_gpay_adjs)
top_100_adjs = df_phonepe_in_gpay_adjs.adjs.value_counts()[0:100]


df_phonepe_in_gpay_verbs = pd.DataFrame({'verbs':verbs})
df_phonepe_in_gpay_verbs = df_phonepe_in_gpay_verbs.verbs.apply(lambda x: x.lower())
df_phonepe_in_gpay_verbs = pd.DataFrame(df_phonepe_in_gpay_verbs)
top_100_verbs = df_phonepe_in_gpay_verbs.verbs.value_counts()[0:100]



## code to check the keywords which can enfluence the decision of the sentiment
## df_check is the temporary dataset to analyse the influenceing keyword and misclassification chances

df_check = pd.DataFrame()
df_check['Reviews'] = df_phonepe_in_paytm['Reviews'].apply(lambda x: x.lower() if ('better than other' in x.lower())  else np.nan)
df_check.dropna(inplace=True) 

analyzer = SentimentIntensityAnalyzer()
df_check['scores'] = df_check.Reviews.apply(analyzer.polarity_scores)
df_check['compound_score_label'] = df_check.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )




################## we are going to use vader for classifing positive and negative reviews and sentiment analysis
## vader analyser
analyzer = SentimentIntensityAnalyzer()
df_phonepe_in_paytm['scores'] = df_phonepe_in_paytm.Reviews.apply(analyzer.polarity_scores)
df_phonepe_in_paytm['compound_score_label'] = df_phonepe_in_paytm.scores.apply(lambda x : 'pos' if x['compound']>0 else 'neg' if x['compound']<0 else 'neutral' )
df_phonepe_in_paytm.compound_score_label.value_counts()

'''
pos        815
neg        629
neutral    278
'''
## word 'better than phone' in reviews refers to google pay and paytm not phone pe.
## though it is a positive word, the review still can be negative in nature
## we removing 'better than phone' from all selected reviews.

def keyword_analyzer(x):
    if ('worst' in x) or ('pathetic' in x) or ('unable' in x) or ('slower than other' in x) or ('better than phone' in x) or ('best than phone' in x) or ('not better than paytm' in x) or ('waste than patm' in x):
        return 'neg'
    elif ('not better than phone' in x) or ('better than paytm' in x) or ('best than paytm' in x) or ('good than paytm' in x) or ('faster than paytm' in x) or ('better than other' in x):
        return 'pos'
    

df_phonepe_in_paytm['temp'] = df_phonepe_in_paytm['Reviews'].apply(keyword_analyzer)


for index in df_phonepe_in_paytm.index:
    if df_phonepe_in_paytm['temp'][index] != None:
        df_phonepe_in_paytm.compound_score_label[index]=df_phonepe_in_paytm['temp'][index]
        
df_phonepe_in_paytm.drop(columns=['temp'],inplace=True)
df_phonepe_in_paytm.compound_score_label.value_counts()
ax = sns.countplot(x='compound_score_label', data=df_phonepe_in_paytm)
plt.title('phonepe comment in paytm dataset')

'''
pos        415
neg        230
neutral     84

there were total 415 reviews of phonepe in which response for phonepe is positive against paytm 
there were total 230 reviews of phonepe in which response for phonepe is negative against paytm

there are total 84 neutral reviews, most of them are advertigements
'''

