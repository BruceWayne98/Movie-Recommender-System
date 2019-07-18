#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing

# In[1]:


import pandas as pd
# from bs4 import BeautifulSoup
# import requests
# from requests import get


# In[2]:


# url= 'https://www.imdb.com/chart/top'
# response= requests.get(url)
# page=response.content
# soup = BeautifulSoup(page, 'html.parser')
# No need to run this coz file is present offline, skip
# class_=soup.find_all(name='div',attrs={'class':'wlb_ribbon'})
# movie_ids=[c['data-tconst'] for c in class_]
# movie_info=[[] for i in range(len(movie_ids))]

# for i in range(350):
#     url='http://www.omdbapi.com/?i='
#     r=requests.get(url+movie_ids[i]+"&apikey=de12b217").json()
#     for a in r.keys():
#         movie_info[i].append(r[a])
        
# df=pd.DataFrame(movie_info,columns=r.keys())
# df.to_csv("movies.csv", sep=',')


# In[3]:


#df.to_csv('movies.csv',sep='\t')
df=pd.read_csv('movies.csv', sep=',',index_col=0)


# In[4]:


df.head()


# ## Feature Extraction

# In[5]:


import math
from textblob import TextBlob as tb


# In[6]:


def tf(word, sent):
    return sent.words.count(word) / len(sent.words)
def n_containing(word, sent_list):
    return sum(1 for sent in sent_list if word in sent.words)
def idf(word, sent_list):
    return math.log(len(sent_list) / (1 + n_containing(word, sent_list)))
def tfidf(word, sent, sent_list):
    return tf(word, sent) * idf(word, sent_list)


# In[7]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk


# In[8]:


stop_words=set(stopwords.words('english'))
print(stop_words)


# In[9]:


ini_blob_list=[tb(plot) for plot in list(df['Plot'])]
blob_list=[]
for i in ini_blob_list:
    item= list(i.words)
    new_item=[word.lower() for word in item if not word in stop_words]
    new_word=''
    for j in new_item:
        new_word+=j
        new_word+=' '
    new_word.strip(' ')
    #print(new_word)
    blob_list.append(tb(new_word))


# In[10]:


blob_list[0].words


# In[11]:


pos_tag_list=[]
for i in blob_list:
    pos_tag_list.append(nltk.pos_tag(i.words))


# In[12]:


pos_tag_list[0]


# ## Feature Refinement

# In[13]:


from nltk.stem import WordNetLemmatizer as lemma


# In[14]:


lem = lemma()
feature_list=list()
for i in range(len(pos_tag_list)):
    temp=[]
    for word, tag in pos_tag_list[i]:
        if tag.startswith('NN'):
             temp.append(lem.lemmatize(word))
    feature_list.append(temp)


# In[15]:


feature_list[0]


# In[16]:


# top_dict=[]


# In[17]:



# for i, blob in enumerate(blob_list):
#     scores = {word: tfidf(word, blob, blob_list) for word in list(blob.words)}
#     scores= sorted(scores.items(), key= lambda x:x[1] ,reverse=True)
#     #rint(scores)
#     top_words=[w[0] for w in scores if w[1]>0.15]
#     top_dict.append(top_words)


# In[18]:


df['bag']=''


# In[19]:


# top_dict[0]


# In[20]:


# '''
# i=0
# for index, row in df.iterrows():
#     st=''
#     for j in top_dict[i]:
#         st+=j
#         st+=' '
#     st=st.strip(' ')
#     row.loc['bag']=st
#     i+=1


# In[21]:


i=0
for index, row in df.iterrows():
    st=''
    for j in feature_list[i]:
        st+=j
        st+=' '
    st=st.strip(' ')
    df.loc[index,'bag']=st
    i+=1


# In[22]:


for i in range(250):
    print(i,"\t",df['bag'].loc[i])


# ## Categorization

# In[23]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[24]:


count=CountVectorizer()
count_matrix=count.fit_transform(df['bag'])


# In[25]:


# count_matrix=count.fit_transform(list(df['bag']))
# list(df['bag'])
type(df['bag'])


# In[26]:


cosine_sim=cosine_similarity(count_matrix, count_matrix)


# In[27]:


cosine_sim


# In[28]:


indices= pd.Series(df.index)


# ## Result

# In[29]:


def recommend_movie(title, cosine_sim=cosine_sim):
    indices= pd.Series(df.index)
    rec_movies=[]
    movie_id= df.loc[df['Title'] == title].index[0]
    idx= indices[indices == movie_id].index[0]
    score_series= pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1:11].index)
    for ind in top_10_indexes:
        rec_movies.append(df['Title'].iloc[ind])
    return rec_movies


# In[30]:


recommend_movie('Fargo')


# In[31]:


def rec_actor(name):
    movies=[]
    name=name.lower()
    for i in range(250):
        s=df['Actors'].iloc[i].lower()
        if name in s:
            movies.append(df['Title'].iloc[i])
    return movies


# In[32]:


rec_actor('tim robbins')


# In[33]:


# def rec_multi_movie(movies, cosine_sim= cosine_sim):
#     top_10_dict=dict()
#     for movie in movies:
#         movie_id= df.loc[df['Title'] == movie].index[0]
#         idx=indices[indices == movie_id].index[0]
#         score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
#         d=score_series.to_dict()
#         l=list(d)
#     for i in range(1,11):
#             if l[i] not in top_10_dict:
#                 top_10_dict[l[i]]=d[l[i]]
#             else:
#                 if top_10_dict[l[i]] < d[l[i]]:
#                     top_10_dict[l[i]]=d[l[i]]
#     top_10_keys = top_10_dict.keys()
#     top_10_movies = list()
#     for ind in top_10_keys:
#         top_10_movies.append(df['Title'].iloc[ind])
#     return top_10_movies


# In[32]:


# rec_multi_movie(['The Pianist', "Schindler's List"])


# In[37]:


stop_words.add('a')
stop_words.add('A')
plots = list(df["Plot"])
common_plot = ""


# In[39]:


def multi_m(movies):
    multi_movie_plots=dict()
    for movie in movies:
        id= df.loc[df['Title'] == movie].index[0]
     #   movie_plot = plots[movie_id]
        pl_list = list(plots[id].split(" "))
        pl=set([word.lower() for word in pl_list if not word in stop_words])
        multi_movie_plots[id] = pl
    global common_plot
    common_plot = ""
    common_plot = ' '.join(list(frozenset().union(*multi_movie_plots.values())))
    new_row = ["common","","","","","","","","",common_plot,"","","","","","","","","","","","","","","",common_plot]
    df.loc[250] = new_row
    count=CountVectorizer()
    count_matrix=count.fit_transform(df['bag'])
    cosine_sim=cosine_similarity(count_matrix, count_matrix)
    return recommend_movie("common",cosine_sim)[2:]


# In[40]:


multi_m(['The Pianist', "Schindler's List"])

