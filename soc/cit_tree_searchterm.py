#%%

import pandas as pd
import xarray as xr
import os
import sqlite3
import networkx as nx

import matplotlib.pyplot as plt
from nlp_utils.io import load_df_semantic
import nlp_utils as nu

from nlp_utils.citation import gen_citation_tree, trim_graph_fraction, get_frac_connected, trim_graph_size

db_folder = r'E:\\'

con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()


#%%
def set_round(G, round):
    """
    Set 'round' attribute of each node if it doesn't already exist
    """
    for node in G.nodes:
        if 'round' not in G.nodes[node]:
            G.nodes[node]['round'] = round

regex = '%natural language processing%'
ids = nu.io.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e6))

G = nx.Graph()
G.add_nodes_from(ids, round=0)

## Initial internal citation trim
G = gen_citation_tree(G, con, cit_field='inCitations', add_new=False)
print("size of citation tree: {}".format(len(G.nodes)))
G = trim_graph_size(G, 100)


#Grow citation graph including external citations
for i, n_max in enumerate([500,5000,50000]):
    print('Round {}'.format(i+1))
    G = gen_citation_tree(G, con, cit_field='both', add_new=True)
    print("size of final citation tree: {}".format(len(G.nodes)))
    G = trim_graph_size(G, n_max)
    set_round(G, i+1)

df = load_df_semantic(con, G.nodes)
G.remove_nodes_from([id for id in G.nodes if id not in df.index])


#%%

# from bokeh.plotting import figure, from_networkx
# from bokeh.io import output_notebook, show, output_file
# from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
#                           MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,Legend,LegendItem)
# from bokeh.palettes import Spectral5

# cmap = Spectral5

# for node in G.nodes:
#     # if node in df_2.index:
#     G.nodes[node]['title']= df.loc[node]['title']
#     G.nodes[node]['cit_round'] = cmap[G.nodes[node]['round']]

# plot = figure(plot_width=800, plot_height=800)
# graph = from_networkx(G, nx.spring_layout)

# graph.node_renderer.glyph = Circle(size=15, fill_color='cit_round')
# graph.node_renderer.selection_glyph = Circle(size=15, fill_color=cmap[2])
# graph.node_renderer.hover_glyph = Circle(size=15, fill_color=cmap[1])

# graph.edge_renderer.glyph = MultiLine(line_color='black', line_alpha=0.4, line_width=1)
# graph.edge_renderer.selection_glyph = MultiLine(line_color=cmap[2], line_width=1)
# graph.edge_renderer.hover_glyph = MultiLine(line_color=cmap[1], line_width=1)

# graph.selection_policy = NodesAndLinkedEdges()

# plot.add_tools(HoverTool(tooltips=[('title', '@title')]), TapTool(), BoxSelectTool())
# plot.renderers.append(graph)

# output_notebook()
# show(plot)



#%%
print("Topic Modeling")
df_tm = load_df_semantic(con, G.nodes)
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

gen_lit_tw = pd.read_csv('data/gen_literature_top_words.csv',index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values

corex_anchors = []
fixed_bigrams = nu.corex_utils.anchors_to_fixed_bigrams(corex_anchors)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct

pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

X = pipeline.fit_transform(texts)
feature_names = pipeline['vectorizer'].get_feature_names()

topic_model = ct.Corex(n_hidden=50, seed=42)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

import _pickle as cPickle
#Save model
model_name =  'mod_cit_tree.pkl' 
output_folder = r'C:\Users\aspit\Git\NLP-Semantic\soc\output'
cPickle.dump(topic_model, open(os.path.join(output_folder, model_name), 'wb'))

#%%

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['year'], norm_each_topic=False)


from importlib import reload
reload(nu.plot)

highlight_topics = ['topic_' + str(i) for i in range(len(corex_anchors))]

year_range_fit = slice(2015,2020)
year_range_plot = slice(1990,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=30, highlight_topics=highlight_topics)





#%%
# from nlp_utils.gensim_utils import basic_gensim_lda
# from nlp_utils import gensim_utils

# pipeline = Pipeline([
#     ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
#     ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
#     # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# ])

# texts_bigram = pipeline.fit_transform(texts)

# n_topics = 50
# alpha = 1/n_topics

# lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
# id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)


# lda_model.texts_bigram = texts_bigram
# lda_model.id2word = id2word
# lda_model.data_words = data_words
# lda_model.idx = df_tm.index.values
# lda_model.save(r'C:\Users\aspit\Git\NLP-Semantic\soc\output\ldamod_cit_tree.lda')

