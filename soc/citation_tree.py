#%%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

from nlp_utils.fileio import load_df_semantic

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'soc.db')
con = sqlite3.connect(db_path)

all_dois = pd.read_csv('data/tech_doi.csv')
input_dois = all_dois['general'].dropna()

df = load_df_semantic(con, input_dois, cust_idx_name='doi')

for doi in input_dois:
    if doi not in df['doi'].values:
        print("Could not find doi: " + str(doi))

df

#%%

from nlp_utils.citation import get_citations, build_citation_community
df_temp = build_citation_community(df, con, n_iter=2, frac_keep_factor=0.5, n_initial_trim=200)

#%%

df_temp['outCitations'].apply(len).hist(bins=100)
plt.yscale('log')
plt.xscale('log')

#%%

df_temp.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()

#%%

#how many papers have the term 'review'
review_papers = (df_temp['title'] + df_temp['paperAbstract']).str.contains('review')
review_papers.value_counts()

#%%

df_review = df_temp.where(review_papers==True).dropna(how='all')
df_review['outCitations'].apply(len).hist(bins=1000, alpha=0.5)
df_not_review = df_temp.where(review_papers==False).dropna(how='all')
df_not_review['outCitations'].apply(len).hist(bins=1000, alpha=0.5 )

plt.yscale('log')
plt.xscale('log')
#%%

df_not_review.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()

#%%

df_not_review = df_not_review.where(df_not_review['outCitations'].apply(len) < 100).dropna(how='all')

df_not_review.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()


#%%
from nlp_utils.text_process import TextNormalizer

df_tm = df_temp

# The text we will analyze is words in both the title and abstract concatenated. 
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

text_normalizer = TextNormalizer()

gen_lit_tw = pd.read_csv('data/gen_literature_top_words.csv',index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values

text_normalizer.post_stopwords = gen_lit_remove

texts_out = list(text_normalizer.transform(texts))

#%%


from importlib import reload
from nlp_utils import gensim_utils
reload(gensim_utils)

from nlp_utils.gensim_utils import Gensim_Bigram_Transformer

fixed_bigrams = ['heat_pump', 'pump_thermal'] #TODO: perhaps didn't have to do this, thought it was pumped_thermal before. 
lda_kwargs = {'threshold':20, 'min_count':10}

# texts_bigram = gensim_bigram(texts_out, lda_kwargs, fixed_bigrams)

bigram_trans = Gensim_Bigram_Transformer()

texts_bigram = bigram_trans.fit(texts_out, lda_kwargs, fixed_bigrams).transform(texts_out)
docs_bigram = [" ".join(t) for t in texts_bigram]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=None, min_df=2, max_df = 0.9)
X = vectorizer.fit_transform(docs_bigram)
feature_names = vectorizer.get_feature_names()


# %%
from corextopic import corextopic as ct
from nlp_utils import corex_utils

# topic_model = ct.Corex(n_hidden=50)  # Define the number of latent (hidden) topics to use.
# topic_model.fit(X, words=feature_names, docs=docs.index, anchors=None, anchor_strength=5)

topic_model = ct.Corex(n_hidden=30)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=['compress_air', 'pump_thermal', 'lithium_ion'], anchor_strength=5)


topic_names = ['topic_' + str(i) for i in range(topic_model.n_hidden)]

s_topic_words = corex_utils.get_s_topic_words(topic_model, topic_names)
s_topic_words

#%%

import nlp_utils.common as nu_common 
doc_topic_probs = topic_model.p_y_given_x
df_doc_topic_probs = pd.DataFrame(doc_topic_probs, index=df_tm.index , columns=topic_names)

s_year = pd.Series(df_tm['year'], index=df_tm.index)

df_topicsyear = nu_common.calc_topics_year(df_doc_topic_probs, s_year, norm_each_topic=False)
df_topicsyear.index = df_topicsyear.index.astype(int)
df_topicsyear = df_topicsyear.sort_index()

#%%
from nlp_utils.plot import top_slopes_plot

topic_strs = s_topic_words

year_range_fit = slice(2015,2020)
year_range_plot = slice(2000,2020)

top_slopes_plot(df_topicsyear.loc[year_range_plot], topic_strs, year_range_fit, n_plots=20)
#%%
top_slopes_plot(df_topicsyear.loc[year_range_plot], topic_strs, year_range_fit, n_plots=10, ascending=True)
#%%

topic_num = 15
topic_str = 'topic_' + str(topic_num)

print(topic_str)
print(s_topic_words[topic_str])

top_docs = topic_model.get_top_docs()[topic_num]
top_docs

print("-"+"\n-".join(df_tm['title'].loc[[t[0] for t in top_docs]]))

#%%

###LDA####

from nlp_utils.gensim_utils import basic_gensim_lda
from nlp_utils import gensim_utils

n_topics = 50
alpha = 1/n_topics

lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)

# %%

df_topickeywords, doc_topic_probs = gensim_utils.gensim_topic_info(lda_model, data_words, id2word)

df_doc_topic_probs = pd.DataFrame(doc_topic_probs, columns=df_topickeywords.index)
df_doc_topic_probs.index = df_tm.index.values

#%%

s_topic_words = df_topickeywords.apply(" ".join, axis=1)
s_year = pd.Series(df_tm['year'], index=df_tm.index.values)

df_topicsyear = nu_common.calc_topics_year(df_doc_topic_probs, s_year, norm_each_topic=False)
df_topicsyear.index = df_topicsyear.index.astype(int)
df_topicsyear = df_topicsyear.sort_index()


# %%

year_range_fit = slice(2015,2020)
year_range_plot = slice(2000,2020)

top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=20)
#%%

top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=10, ascending=True)
# %%
topic = 'topic_44'

top_docs = df_doc_topic_probs[topic].sort_values(ascending=False)
top_docs = top_docs.index[0:10]
# df_tm.loc[top_papers[0:10].index]

disp_strs = df_tm['title'] + ' (' + df_tm['year'].astype(str) + ')' 

print("-"+"\n-".join(disp_strs.loc[top_docs].values))