#%%

import pandas as pd
import xarray as xr
import os
import sqlite3
import networkx as nx

import matplotlib.pyplot as plt
from nlp_utils.io import load_df_semantic
import nlp_utils as nu

from nlp_utils.citation import gen_citation_tree, trim_graph, get_frac_connected

# db_folder = r'/media/lee/Shared Storage'
db_folder = r'E:\\'

# %%

con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

all_dois = pd.read_csv('data/tech_doi.csv')
input_dois = all_dois['general'].dropna()
df = nu.io.load_df_semantic(con, input_dois, cust_idx_name='doi')

G = nx.Graph()
G.add_nodes_from(df.index.values, round=0)

for round in [1, 2]:
    df = load_df_semantic(con, G.nodes)
    G = gen_citation_tree(G, df)
    G = get_frac_connected(G, con)
    G = trim_graph(G, frac_keep_factor=1, min_connect=1)

    for node in G.nodes:
        if 'round' not in G.nodes[node]:
            G.nodes[node]['round'] = round

    G = nx.Graph(G)

df = load_df_semantic(con, G.nodes)
con.close()
#%%


from bokeh.plotting import figure, from_networkx
from bokeh.io import output_notebook, show, output_file
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,)
from bokeh.palettes import Spectral4

for node in G.nodes:
    # if node in df_2.index:
    G.nodes[node]['title']= df.loc[node]['title']
    G.nodes[node]['round'] = Spectral4[G.nodes[node]['round']]


plot = figure(plot_width=800, plot_height=800)
graph = from_networkx(G, nx.spring_layout)

graph.node_renderer.glyph = Circle(size=15, fill_color='round')
graph.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
graph.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

graph.edge_renderer.glyph = MultiLine(line_color='black', line_alpha=0.4, line_width=1)
graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=1)
graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=1)


graph.selection_policy = NodesAndLinkedEdges()
# graph.inspection_policy = EdgesAndLinkedNodes()

plot.add_tools(HoverTool(tooltips=[('title', '@title')]), TapTool(), BoxSelectTool())
plot.renderers.append(graph)

output_notebook()
show(plot)

# output_file('test.html')
# show(plot)

#%%
df_com = df

docs = df_com['title'] + ' ' + df_com['paperAbstract']
texts = docs.apply(str.split)

gen_lit_tw = pd.read_csv('data/gen_literature_top_words.csv',index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values

corex_anchors = [['fuel_cel', 'electrolyz'], ['pump_thermal', 'heat_pump'], 'li_ion']
fixed_bigrams = nu.corex_utils.anchors_to_fixed_bigrams(corex_anchors)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct

pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    ('vectorizer', CountVectorizer(max_features=None, min_df=2, max_df = 0.9, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

X = pipeline.fit_transform(texts)
feature_names = pipeline['vectorizer'].get_feature_names()

topic_model = ct.Corex(n_hidden=30, seed=42)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_com.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_com['year'], norm_each_topic=False)
# %%

year_range_fit = slice(2015,2020)
year_range_plot = slice(2000,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=30)


#%%

