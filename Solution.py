import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import re
df = pd.read_csv('new.csv', usecols=['URL_ID','URL','TEXT'])
stopWordsFile = 'sm.txt'

with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []
lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')

file = open('negative-words.txt', 'r')
neg_words = file.read().split()
file = open('positive-words.txt', 'r')
pos_words = file.read().split()

def text_prep(x):
     corp = str(x).lower() 
     corp = re.sub('[^a-zA-Z]+',' ', corp).strip() 
     tokens = word_tokenize(corp)
     words = [t for t in tokens if t not in stop_words]
     lemmatize = [lemma.lemmatize(w) for w in words]
     return lemmatize


def sentance(text):
    sentence_list = sent_tokenize(text)
    totalSentences = len(sentence_list)
    return totalSentences

def syll(s):
    vow=('a', 'e', 'i', 'o', 'u')
    c=0
    l=0
    for i in s:
        if i in vow:
            c+=1
    arr=s.split()
    for i in arr:
        if i[-2:]=='ed' or i[-2:]=='es':
            l+=1
    #print(arr)
    return(c-l)


def per(s):
    c=0
    l=('I','we','my','ours','us')
    arr=s.split()
    for i in arr:
        if i in l and i!='US':
            c+=1
    return c

def tnw(s):
    c=0
    for i in s:
        c+=1
    return c


    
preprocess_tag = [text_prep(i) for i in df['TEXT']]
df['COMPLEX WORD COUNT'] = preprocess_tag


avg = df['COMPLEX WORD COUNT'].map(lambda x: len(x))
preprocess_tag = [text_prep(i) for i in df['TEXT']]
df['COMPLEX WORD COUNT'] = preprocess_tag
avg = df['COMPLEX WORD COUNT'].map(lambda x: len(x))

# Positive Score
num_pos = df['COMPLEX WORD COUNT'].map(lambda x: len([i for i in x if i in pos_words]))
df['POSITIVE SCORE'] = num_pos

# Negative Score
num_neg = df['COMPLEX WORD COUNT'].map(lambda x: len([i for i in x if i in neg_words]))
df['NEGATIVE SCORE'] = num_neg

# Polarity Score
df['POLARITY SCHORE'] = (df['POSITIVE SCORE']-df['NEGATIVE SCORE'])/ ((df['POSITIVE SCORE']+df['NEGATIVE SCORE']) + 0.000001)

# Subjective Score
df['SUBJECTIVE SCORE'] = (df['POSITIVE SCORE'] + df['NEGATIVE SCORE'])/ ((avg) + 0.000001)
tw = df['TEXT'].str.split().str.len()

# Average sentance Length
sln= [sentance(i) for i in df['TEXT']]
df['AVERAGE SENTANCE LEN'] = sln

# Persentage of Complex Word
# need to change
df['PERSENTAGE OF COMPLEX WORDS'] = tw / df['AVERAGE SENTANCE LEN']

# Fog Index
df['FOG INDEX'] = 0.4 * (df['AVERAGE SENTANCE LEN'] + df['PERSENTAGE OF COMPLEX WORDS'])

# Average numbe of Words per SEntance
df['AVG NUMBER OF WORDS PER SENTENCE'] = len(str(df['COMPLEX WORD COUNT'])) / df['AVERAGE SENTANCE LEN']

# Complex Word Count
df['COMPLEX WORD COUNT1']= len(str(df['COMPLEX WORD COUNT']))

# Word Count
df['WORD COUNT']=tw

# Syllable per word
syl=[syll(i) for i in df['TEXT']]
df['SYLLABLE PER WORD']=syl

# Personal pronouns
pers=[per(i) for i in df['TEXT']]
df['PERSONAL PRONOUNS']=pers

# Average WOrd Length
tw=[tnw(i) for i in df['TEXT']]
df['AVG WORD LENGTH']=tw/df['WORD COUNT']
df.head()
df.to_csv('Output.csv')