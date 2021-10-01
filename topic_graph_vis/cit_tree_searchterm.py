#%%

from bokeh.models.renderers import GraphRenderer
from bokeh.models.sources import ColumnDataSource
import pandas as pd
import xarray as xr
import os
import sqlite3
import networkx as nx

import matplotlib.pyplot as plt
from nlp_utils.io import load_df_semantic
import nlp_utils as nu

from nlp_utils.citation import gen_citation_tree, trim_graph_fraction, get_frac_connected, trim_graph_num_edges, trim_graph_size

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
    return G

regex = '%energy storage%'
ids = nu.io.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(5e6), output_limit=1000)

#%%

def trim_graph(G, n_max):
    G = trim_graph_size(G, n_max*2)
    G = trim_graph_num_edges(G, 1)
    G = get_frac_connected(G, con)
    G = trim_graph_fraction(G, n_max)
    G = trim_graph_num_edges(G, 1)
    return G

all_Gs = []

G = nx.Graph()
G.add_nodes_from(ids, round=0)

## Initial internal citation trim
G = gen_citation_tree(G, con, cit_field='inCitations', add_new=False)
print("size of citation tree: {}".format(len(G.nodes)))

G = nu.citation.trim_connected_components(G, 10)
G = trim_graph(G, 100)

all_Gs.append(G.copy())


#%%
#Grow citation graph including external citations
for i, n_max in enumerate([300,1000,3000,10000]):
    print('Round {}'.format(i+1))
    G = gen_citation_tree(G, con, cit_field='both', add_new=True)
    print("size of final citation tree: {}".format(len(G.nodes)))
    G = trim_graph(G, n_max)

    G = nu.citation.trim_connected_components(G, 1)
    set_round(G, i+1)

    all_Gs.append(G.copy())


print("Loading final database")
df = load_df_semantic(con, G.nodes)

print("removing nodes not in database")
G.remove_nodes_from([id for id in G.nodes if id not in df.index])

nx.write_gexf(G, os.path.join('output','G_cit_tree.gexf'))
#%%

from fa2 import ForceAtlas2
forceatlas2 = ForceAtlas2()
#%%

rounds_plot = [0,1,2,3,4]

fig, axes = plt.subplots(1,len(rounds_plot), figsize=(15,5))

for round in rounds_plot:
    G_plot = all_Gs[round]
    # pos = nx.spring_layout(G_plot)
    pos = forceatlas2.forceatlas2_networkx_layout(G_plot, pos=None, iterations=10)
    nx.draw_networkx_nodes(G_plot, pos, ax=axes[round], node_size=10)
    nx.draw_networkx_edges(G_plot, pos, ax=axes[round],width=0.1)

    axes[round].set_title('Round {}'.format(round))


plt.savefig('output/cit_tree_rounds.png')

#%%

from bokeh.plotting import figure, from_networkx
from bokeh.io import output_notebook, show, output_file
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,Legend,LegendItem, ColumnDataSource)
from bokeh.palettes import Spectral4, Spectral6

cmap = Spectral6

rounds_plot = [0,1,2,3,4]
nodes_plot = [node for node in G.nodes if G.nodes[node]['round'] in rounds_plot]
G_plot = G.subgraph(nodes_plot)
G_plot = nx.Graph(G_plot)
# G_plot.remove_edges_from(G.edges) # Speed
print(len(G_plot.nodes))

plot = figure(plot_width=800, plot_height=600, output_backend='webgl')

#Dummy markers for legend, doesn't appear to be a way to have a colored legend with networx plot...
rounds = set(G_plot.nodes[node]['round'] for node in G_plot.nodes)
source = ColumnDataSource(dict(
    x = [0]*len(rounds),
    y = [0]*len(rounds),
    color = [cmap[r] for r in rounds],
    label=["Round {}".format(r) for r in rounds]
))

plot.circle(source=source, legend_group = 'label', color = 'color', visible=False)

# Networkx plot
for node in G_plot.nodes:
    # if node in df_2.index:
    G_plot.nodes[node]['title']= df.loc[node]['title']
    G_plot.nodes[node]['cit_round'] = cmap[G_plot.nodes[node]['round']]


pos = forceatlas2.forceatlas2_networkx_layout(G_plot, pos=None, iterations=10)
source = ColumnDataSource(dict(
x = [pos[key][0] for key in pos.keys()],
y = [pos[key][1] for key in pos.keys()],
color = [cmap[G_plot.nodes[node]['round']] for node in pos.keys()],
title = [G_plot.nodes[node]['title'] for node in pos.keys()],
))

plot.circle(source = source, color='color', radius = 3, radius_units='screen')

plot.add_tools(HoverTool(tooltips=[('title', '@title')]))

## from_networkx method doesn't seem to work with webgl
# graph = from_networkx(G_plot, lambda G: forceatlas2.forceatlas2_networkx_layout(G, iterations=10))
# graph.node_renderer.glyph = Circle(fill_color='cit_round')
# # graph.node_renderer.hover_glyph = Circle(size=15, fill_color='gray')
# # # graph.edge_renderer.glyph = MultiLine(line_color='black', line_alpha=0.4, line_width=1)
# graph.edge_renderer.visible = False
# plot.renderers.append(graph)

# output_notebook()
# show(plot)

output_file(filename='output/cit_tree_bokeh.html', title = 'Initial Citation Tree Rounds', )
show(plot)

#%%
print("Corex Topic Modeling")
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

topic_model = ct.Corex(n_hidden=30, seed=42)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

import _pickle as cPickle
#Save model
if not os.path.exists('models'): os.mkdir('models')
cPickle.dump(topic_model, open(os.path.join('models', 'mod_cit_tree.pkl'), 'wb'))

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

plt.savefig('output/top_slopes_plot.png')



#%%

# print("LDA Topic Modeling")
# from nlp_utils.gensim_utils import basic_gensim_lda
# from nlp_utils import gensim_utils

# pipeline = Pipeline([
#     ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
#     ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
#     # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# ])

# texts_bigram = pipeline.fit_transform(texts)

# n_topics = 30
# alpha = 1/n_topics

# lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
# id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)


# lda_model.texts_bigram = texts_bigram
# lda_model.id2word = id2word
# lda_model.data_words = data_words
# lda_model.idx = df_tm.index.values
# lda_model.save(r'C:\Users\aspit\Git\NLP-Semantic\soc\output\ldamod_cit_tree.lda')

