# # Topic Modeling for Finding Table Titles Relevant to Drug-Adverse Events
#
# - The following codes is implemented for topic modeling through LDA in attempt to find document samples(table title data) relevant to drug and adverse events.
# - Topic modeling is one of statistical modeling for discovering the abstract 'topics' that occur in a collection of document.
#  - An example of topic model is used to classify text in a document to a particular topic
#  - it builds a topic per document model and words per topic model, modeled as Dirichlet distributions
# - Following urls and research articles were referenced.
#  - https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
#  - http://www.engear.net/wp/topic-modeling-gensimpython/
#  - https://markroxor.github.io/gensim/static/notebooks/lda_training_tips.html
#  - https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#14computemodelperplexityandcoherencescore
#  - https://markroxor.github.io/gensim/static/notebooks/lda_training_tips.html
#  - http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb
#  - http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276
#  - https://ldavis.cpsievert.me/reviews/reviews.html

# ### Latent Dirichlet Allocation

import pandas as pd
import numpy as np
import gensim

from input_util import get_input_text
from preprocessing_util import Preprocessing_util


# #### Data Retrival
documents = get_input_text()

#process the text, save the results as processed_docs
processed_docs = pd.DataFrame()
processed_docs = pd.concat([documents.id, documents.title.map(preprocess)], axis=1)
processed_docs[:5]

preprocessing_util = Preprocessing_util(processed_docs)

processed_bigram = n_gram_lemmatization(make_bigram(processed_docs.title))
processed_trigram = n_gram_lemmatization(make_trigram(processed_docs.title))
print(str(len(processed_bigram)))
print(str(len(processed_trigram)))
processed_docs['bigram'] = processed_bigram
processed_docs['trigram'] = processed_trigram
processed_docs.head(10)


# ### bag of words on the data set
# - create a dictionary from 'processed_docs' containing the number of times a word appears in the training set

# In[13]:


#############
dictionary_made_by = processed_trigram
dictionary_made_by_str = 'trigram'

dictionary = gensim.corpora.Dictionary(dictionary_made_by)
print('size of the dictionary %d' %len(dictionary))
max_dic_size = len(dictionary)
test_dic_sizes = [max_dic_size, int(max_dic_size*2/3), int(max_dic_size/2), 10000]

# count = 0
# for k, v in dictionary.iteritems():
#     if v.find('_')>=0:
#         print(k,v)
#     count += 1
#     if count > 200:
#         break
#     if v.find('tion')>=0:
#         print(k,v)


# In[14]:


# filter out tokens that appear in less than 0.05 documents
# or more than 0.5 documents
# after above, keep only the first 40000 most frequent tokens.

#############
dict_size = test_dic_sizes[1]
dictionary.filter_extremes(no_below=0.05, no_above=0.5, keep_n=dict_size)


# In[15]:


# vectorize
# Bag-of-words representation of the documents
bow_corpus = [dictionary.doc2bow(doc) for doc in dictionary_made_by]

bow_doc_100 = bow_corpus[100]
for i in range(len(bow_doc_100)):
    print("Word {} (\"{}\") appears {} times.".format(bow_doc_100[i][0],
                                                     dictionary[bow_doc_100[i][0]],
                                                     bow_doc_100[i][1]))

processed_docs['bow_corpus'] = bow_corpus
processed_docs.head(10)


# #### TF-IDF
# - tf-idf model on bow_corpus
#
# - tf = count(word, document) / len(document)
# - idf = log( len(collection) / count(document_containing_term, collection)
# - tf-idf = tf * idf

# In[16]:


from gensim import corpora, models
from pprint import pprint

tfidf_mod = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf_mod[bow_corpus]

processed_docs['tfidf_corpus'] = tfidf_corpus
tfidf_doc_100 = tfidf_corpus[100]
for i in range(len(tfidf_doc_100)):
    print("Word {} (\"{}\") appears {} times.".format(tfidf_doc_100[i][0],
                                                     dictionary[tfidf_doc_100[i][0]],
                                                     tfidf_doc_100[i][1]))


# In[17]:


print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents bow corpus: %d' % len(bow_corpus))
print('Number of documents tfidf: %d' % len(tfidf_corpus))


# #### Optimal number of topics
# - build many LDA models with different values of number of topics and pick the one that gives the highest coherence value based on corpus
# - 토픽 모델링 결과로 나온 주제들에 대해 각각의 주제에서 상위 N개의 단어를 뽑습니다. 모델링이 잘 되었을수록 한 주제 안에는 의미론적으로 유사한 단어가 많이 모여있게 마련입니다. 따라서 상위 단어 간의 유사도를 계산하면 실제로 해당 주제가 의미론적으로 일치하는 단어들끼리 모여있는지 알 수 있습니다.

# In[18]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    compute c_v coherence for various number of topics

    parameters:
    dictionary : gensim dictionary
    corpus : gensim corpus
    texts : list of input texts
    limit : max number of topics

    returns:
    model_list : list of LDA topic models
    coherence_values : coherence value
    """

    coherenece_values = []
    u_mass_coherenece_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus,                                                  num_topics=num_topics, id2word=dictionary, workers=8, iterations=60)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        u_mass_coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')

        coherenece_values.append(coherence_model.get_coherence())
        u_mass_coherenece_values.append(u_mass_coherence_model.get_coherence())

    return model_list, coherenece_values, u_mass_coherenece_values

def compute_coherence_values_tfidf(dictionary, corpus, texts, limit, start=2, step=3):
    """
    compute c_v coherence for various number of topics

    parameters:
    dictionary : gensim dictionary
    corpus : gensim corpus
    texts : list of input texts
    limit : max number of topics

    returns:
    model_list : list of LDA topic models
    coherence_values : coherence value
    """

    coherenece_values = []
    u_mass_coherenece_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(tfidf_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=8, iterations=60)

        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        u_mass_coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')

        coherenece_values.append(coherence_model.get_coherence())
        u_mass_coherenece_values.append(u_mass_coherence_model.get_coherence())

    return model_list, coherenece_values, u_mass_coherenece_values


# In[19]:


# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = './mallet-2.0.8/bin/mallet'
from gensim.models import CoherenceModel
limit = 35; start=2; step=1;

model_list, coherence_values, umass_co_val = compute_coherence_values(dictionary=dictionary,                                                                       corpus=bow_corpus,                                                                       texts=processed_trigram,                                                                       start=start,                                                                       limit=limit,                                                                       step=step)

model_list_tfidf, coherence_values_tfidf, umass_co_val_tfidf = compute_coherence_values_tfidf(dictionary=dictionary,                                                                                               corpus=tfidf_corpus,                                                                                               texts=processed_trigram,                                                                                               start=start,                                                                                               limit=limit,                                                                                               step=step)


# In[20]:


model_config='dic_'+dictionary_made_by_str+'_size_'+str(len(dictionary))

import os
directory ='./stat_files/'+model_config
if not os.path.exists(directory):
    os.makedirs(directory)


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


#graph
fig = plt.figure(figsize=(10,5))

x = range(start, limit, step)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, coherence_values)
ax1.set_xlabel("Number of Topics")
ax1.set_ylabel("Coherence score")
#     plt.legend(("coherence_values"), loc="best")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x, umass_co_val)
ax2.set_xlabel("Number of Topics")
ax2.set_ylabel("u mass Coherence score")
#     plt.legend(("u mass coherence_values"), loc="best")

fig.savefig(directory+'/mallet_coherence.png')


# In[23]:


# Print the coherence scores
# one that maximizes the topic coherence

print('Coherence c_v ================')
max_coherence_topic_num = 8 #default
prev_co = 0
first_max_cohe_found = False
for m, cv in zip(x, coherence_values):
    cur_coherence = round(cv, 4)
    print("Num Topics =", m, " has Coherence cv Value of", cur_coherence)
    if (cur_coherence > 0.32) and m > 2:
        if (prev_co <= cur_coherence) and not first_max_cohe_found:
            max_coherence_topic_num = m
        else:
            first_max_cohe_found = True
    prev_co = cur_coherence
print('Best number of topic is : %d' %max_coherence_topic_num)

print('Coherence umass ================')
#where score plateaus안정
for m, um in zip(x, umass_co_val):
    print("Num Topics =", m, " has Coherence umass Value of", round(um, 4))


# In[24]:


#graph
fig = plt.figure(figsize=(10,5))

x = range(start, limit, step)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, coherence_values_tfidf)
ax1.set_xlabel("Number of Topics")
ax1.set_ylabel("Coherence score")
#     plt.legend(("coherence_values"), loc="best")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x, umass_co_val_tfidf)
ax2.set_xlabel("Number of Topics")
ax2.set_ylabel("u mass Coherence score")
#     plt.legend(("u mass coherence_values"), loc="best")

fig.savefig(directory+'/tfidf_coherence.png')


# In[25]:


# Print the coherence scores
# one that maximizes the topic coherence
print('TF-IDF')
print('Coherence c_v ================')

max_coherence_topic_num_tfidf = 8 #default
prev_co = 0
first_max_cohe_found = False
for m, cv in zip(x, coherence_values_tfidf):
    cur_coherence = round(cv, 4)
    print("Num Topics =", m, " has Coherence cv Value of", cur_coherence)
    if (cur_coherence > 0.2):
        if (prev_co <= cur_coherence) and not first_max_cohe_found:
            max_coherence_topic_num_tfidf = m
        else:
            first_max_cohe_found = True
    prev_co = cur_coherence
print('Best number of topic is : %d' %max_coherence_topic_num_tfidf)

print('Coherence umass ================')
for m, um in zip(x, umass_co_val_tfidf):
    print("Num Topics =", m, " has Coherence umass Value of", round(um, 4))


# ### LDA using bag of words
#
# - train LDA using gensim.models.LdaMulticore

# In[26]:


# lda_model_bow = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
# for idx, topic in lda_model_bow.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))

#is mallet better?
from pathlib import Path

dir_to_check = Path(directory+'/data/')
if not dir_to_check.is_dir():
    os.makedirs(directory+'/data/')

optimal_topic_num = max_coherence_topic_num
lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path,                                               corpus=bow_corpus,                                               num_topics=optimal_topic_num,                                               id2word=dictionary,                                            iterations=100,                                             prefix=directory+'/data/')

#show topics
pprint(lda_mallet.show_topics(formatted=False))
# lda_mallet.save('./data/mallet_topics_num_'+str(optimal_topic_num)+'_'+model_config+'.state.gz')
lda_mallet.load_word_topics()


# In[27]:


import math

fig = plt.figure(figsize=(15,15))
fig.suptitle('mallet_topics_num_'+str(optimal_topic_num)+'_'+model_config)

for i in range(optimal_topic_num):
    df=pd.DataFrame(lda_mallet.show_topic(i), columns=['term','prob']).set_index('term')

    axi = fig.add_subplot(math.ceil(optimal_topic_num/2),2,i+1)
    axi.set_title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, palette='Reds_d')
    axi.set_xlabel('probability')

# plt.show()
fig.savefig(directory+'/mallet_topics_num_'+str(optimal_topic_num)+'.png')


# ### LDA using TF-IDF

# In[28]:


lda_model_tfidf = gensim.models.LdaMulticore(tfidf_corpus,                                             num_topics=max_coherence_topic_num_tfidf,                                              id2word=dictionary,                                              passes=2,                                              workers=8,                                            iterations=100)

pprint(lda_model_tfidf.show_topics(formatted=False))


# In[29]:


fig = plt.figure(figsize=(15,15))
fig.suptitle('tfidf_topics_num_'+str(max_coherence_topic_num_tfidf)+'_'+model_config)

for i in range(max_coherence_topic_num_tfidf):
    df=pd.DataFrame(lda_model_tfidf.show_topic(i), columns=['term','prob']).set_index('term')

    axi = fig.add_subplot(math.ceil(max_coherence_topic_num_tfidf/2),2,i+1)
    axi.set_title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, palette='Blues_d')
    axi.set_xlabel('probability')

# plt.show()
fig.savefig(directory+'/tfidf_topics_num_'+str(max_coherence_topic_num_tfidf)+'.png')


# In[30]:


#Check topic distribution of a sample data
for ind, score in sorted(lda_mallet[bow_corpus[45]], key=lambda x: -1*x[1]):
    print('\nScore: {}\t \nTopic: {}'.format(score, lda_mallet.print_topic(ind, 10)))


# In[31]:


import pyLDAvis.gensim

pyLDAvis.enable_notebook()


# In[32]:


import gzip
import os

dataDir = directory+'/data/'

def extract_params(statefile):
    """extract alpha and beta values from MALLET statefile by path to statfile

    Args:
        statefile (str) : Path to statefile produced by MALLET
    Returns:
        tuple: alpha (list), beta
    """

#     with codecs.open(statefile, "r",encoding='utf-8') as state:
    with gzip.open(statefile, 'r') as state:
        params = [x.decode('utf-8').strip() for x in state.readlines()[1:3]]
#         params = [x.strip() for x in state.readlines()[1:3]]
    return (list(params[0].split(':')[1].split(" ")), float(params[1].split(':')[1]))

def state_to_df(statefile):
    """transform state file into pandas dataframe
    the MALLET statefile is tab-separated, and the first two rows contain the alpha and beta parameters

    Args:
        statefile (str): Path to statefile produced by MALLET
    Returns:
        dataframe: topic assignment for each token in each documnet of the model
    """
    return pd.read_csv(statefile,                      compression='gzip',                      sep=' ',                      skiprows=[1,2])


# In[33]:


params = extract_params(os.path.join(dataDir, 'state.mallet.gz'))
alpha = [float(x) for x in params[0][1:]]
beta = params[1]
print("{}, {}".format(alpha, beta))

df = state_to_df(os.path.join(dataDir, 'state.mallet.gz'))
df['type'] = df.type.astype(str)
df[:10]
#doc id, word position index, word index, topic assignmnet


# In[34]:


#to get the length of the document, group by the document id and count the tokens
docs = df.groupby('#doc')['type'].count().reset_index(name='doc_length')
docs[:10]

#get vocab and term frequencies
vocab = df['type'].value_counts().reset_index()
vocab.columns = ['type', 'term_freq']
vocab = vocab.sort_values(by='type', ascending=True)
vocab[:10]


# In[35]:


#matrix file
#need to normalize data so that each row sums to 1
import sklearn.preprocessing

def pivot_and_smooth(df, smooth_values, rows_variable, cols_variable, values_variable):
    """
    modify dataframe into matrix
    Args:
        df (dataframe) :
        smooth_values (float) : value to add to the matrix to account for the priors
        rows_variable (str) : title of rows
        cols_variable (str) : title of columns
        values_variable (str) : values
    Returns:
        dataframe : that has been normalized on the rows.
    """
    matrix = df.pivot(index=rows_variable, columns=cols_variable, values=values_variable).fillna(value=0)
    matrix = matrix.values + smooth_values

    normed = sklearn.preprocessing.normalize(matrix, norm='l1', axis=1)

    return pd.DataFrame(normed)


# In[36]:


#get the number of topic assingments for words in documents
#phi - topic-term matrix and counted the number of times each word was assigned to each topic
#and sorted by alphabetically to match the order of the vocabulary frame

#beta as the smoothign value
phi_df = df.groupby(['topic', 'type'])['type'].count().reset_index(name='token_count')
phi_df = phi_df.sort_values(by='type', ascending=True)
phi_df[:10]

phi = pivot_and_smooth(phi_df, beta, 'topic', 'type', 'token_count')
phi[:10]

#theta document-topic matrix and use alpha as the smoothign value
theta_df = df.groupby(['#doc', 'topic'])['topic'].count().reset_index(name='topic_count')
theta_df[:10]

theta = pivot_and_smooth(theta_df, alpha, '#doc', 'topic', 'topic_count')
theta[:10]


# In[37]:


lda_mallet_data = {
    'topic_term_dists':phi,
    'doc_topic_dists':theta,
    'doc_lengths':list(docs['doc_length']),
    'vocab':list(vocab['type']),
    'term_frequency':list(vocab['term_freq'])
}
mallet_vis_data = pyLDAvis.prepare(**lda_mallet_data)
pyLDAvis.display(mallet_vis_data)
pyLDAvis.save_html(mallet_vis_data, directory+'/mallet_topics_num_'+str(max_coherence_topic_num)+'.html')


# In[38]:


#each bubble = topic, the larger the bubble, the more prevalent is that topic
#good topic = fairly big, non-overlapping bubbles scattered throughout the chart
#model with too many topics = typically have many overlaps, small sized bubbles in one region of the chart
data_tfidf = pyLDAvis.gensim.prepare(lda_model_tfidf, tfidf_corpus, dictionary)
data_tfidf

#bar = salient keywords that form the selected topic
pyLDAvis.display(data_tfidf)
pyLDAvis.save_html(data_tfidf, directory+'/tfidf_topics_num_'+str(max_coherence_topic_num_tfidf)+'.html')


# #### Perplexity and Coherence score
# - model perplexity and topic coherence provide a convenient measure to judge how good a given topic model is

# In[39]:


#compute perplexity : a measure of how good the model is
# lower the better
# print('\nPerplexit: ', lda_mallet.log_perplexity(bow_corpus))
# print('\nPerplexit: ', lda_model_tfidf.log_perplexity(bow_corpus))


# In[40]:


from gensim.models import CoherenceModel

#compute coherence score
coherence_model_lda = CoherenceModel(model=lda_mallet, texts=processed_trigram, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

#compute coherence score
coherence_model_lda_tfidf = CoherenceModel(model=lda_model_tfidf, texts=processed_trigram, dictionary=dictionary, coherence='c_v')
coherence_lda_tfidf = coherence_model_lda_tfidf.get_coherence()
print('\nTF-IDF Coherence Score: ', coherence_lda_tfidf)


# ### Find dominant topic
# - find the most contributed topic

# In[41]:


processed_docs = processed_docs.merge(documents, on='id')
# test = processed_docs[['id', 'title_x', 'bow_corpus', 'title_y']]
# test.to_csv('test.csv')


# In[42]:


processed_docs.head(10)


# In[43]:


import random

def format_topics_sentences(model=lda_mallet,                             corpus=processed_docs,                             texts=dictionary_made_by,                             is_mallet_model=True):
    sent_topic_df = pd.DataFrame()
    if is_mallet_model:
        target_corpus = corpus.bow_corpus
    else:
        target_corpus = corpus.tfidf_corpus

    for i, row in enumerate(model[target_corpus]):
        origin_info = processed_docs.loc[i]
        #get main topic in each document
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j==0: #dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topic_df = sent_topic_df.append(pd.Series([origin_info.id,                                                                int(topic_num),                                                                 round(prop_topic, 4),                                                                 topic_keywords,                                                                origin_info.title_y]), ignore_index=True)
            else:
                break
    sent_topic_df.columns = ['id', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Origin_Text']

    #add original text to the end of the output
#     contents = pd.Series(texts)
    sent_topic_df['Text_Vec'] = pd.Series(texts)#pd.concat([sent_topic_df, contents], axis=1)
#     sent_topic_df
    return (sent_topic_df)


# In[44]:


df_topic_sents_keywords_mallet = format_topics_sentences(model=lda_mallet,                                                   corpus=processed_docs,                                                   texts=dictionary_made_by,                                                  is_mallet_model=True)
df_topic_sents_keywords_mallet.head(20)


# In[45]:


df_topic_sents_keywords_tfidf = format_topics_sentences(model=lda_model_tfidf,                                                   corpus=processed_docs,                                                   texts=dictionary_made_by,                                                  is_mallet_model=False)
df_topic_sents_keywords_tfidf.head(20)


# #### 각 토픽별로 가장 대표적인 문서 찾기

# In[46]:


sent_topics_sorted_df_mallet = pd.DataFrame()
sent_topics_groupby = df_topic_sents_keywords_mallet.groupby('Dominant_Topic')

for i, grp in sent_topics_groupby:
    sent_topics_sorted_df_mallet = pd.concat([sent_topics_sorted_df_mallet,                                              grp.sort_values(['Perc_Contribution'], ascending=[0]).head(10)],                                             axis=0)

sent_topics_sorted_df_mallet.reset_index(drop=True, inplace=True)
sent_topics_sorted_df_mallet.columns = ['id', 'Topic_Num', 'Topic_Perc_Contribu', 'Topic_Keywords', 'Origin_Text', 'Text']
sent_topics_sorted_df_mallet[:10]


# In[47]:


sent_topics_sorted_df_tfidf = pd.DataFrame()
sent_topics_groupby = df_topic_sents_keywords_tfidf.groupby('Dominant_Topic')

for i, grp in sent_topics_groupby:
    sent_topics_sorted_df_tfidf = pd.concat([sent_topics_sorted_df_tfidf,                                              grp.sort_values(['Perc_Contribution'], ascending=[0]).head(10)],                                             axis=0)

sent_topics_sorted_df_tfidf.reset_index(drop=True, inplace=True)
sent_topics_sorted_df_tfidf.columns = ['id', 'Topic_Num', 'Topic_Perc_Contribu', 'Topic_Keywords', 'Origin_Text', 'Text']
sent_topics_sorted_df_tfidf[:10]


# #### 문서 전체적인 토픽 분포

# In[48]:


pd.set_option('display.max_colwidth', -1)

topic_counts = df_topic_sents_keywords_mallet['Dominant_Topic'].value_counts()
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

topic_num_keywords = df_topic_sents_keywords_mallet[['Dominant_Topic', 'Topic_Keywords']]

df_dominant_topics = pd.concat([topic_counts, topic_contribution], axis=1)
df_dominant_topics.reset_index(level=0, inplace=True)
df_dominant_topics.columns = ['Dominant_Topic', 'Num_Documents', 'Perc_Documents']
df_dominant_topics = df_dominant_topics.merge(topic_num_keywords.drop_duplicates(), on='Dominant_Topic')
df_dominant_topics = df_dominant_topics.sort_values('Dominant_Topic')

df_dominant_topics.reset_index(drop = True, inplace = True)
df_dominant_topics.to_html(directory+'/mallet_distribution.html', index=False)
df_dominant_topics.head(optimal_topic_num)


# In[49]:


topic_counts = df_topic_sents_keywords_tfidf['Dominant_Topic'].value_counts()
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

topic_num_keywords = df_topic_sents_keywords_tfidf[['Dominant_Topic', 'Topic_Keywords']]

df_dominant_topics = pd.concat([topic_counts, topic_contribution], axis=1)
df_dominant_topics.reset_index(level=0, inplace=True)
df_dominant_topics.columns = ['Dominant_Topic', 'Num_Documents', 'Perc_Documents']
df_dominant_topics = df_dominant_topics.merge(topic_num_keywords.drop_duplicates(), on='Dominant_Topic')
df_dominant_topics = df_dominant_topics.sort_values('Dominant_Topic')

df_dominant_topics.reset_index(drop = True, inplace = True)
df_dominant_topics.to_html(directory+'/tfidf_distribution.html', index=False)
df_dominant_topics.head(optimal_topic_num)


# - https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

# - https://ai.googleblog.com/2016/12/open-sourcing-embedding-projector-tool.html
# - http://projector.tensorflow.org/
# - http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb#loading

# In[ ]:
