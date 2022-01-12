#%%
import os
import sqlite3
import networkx as nx

import numpy as np
import pandas as pd
import _pickle as cPickle
import matplotlib.pyplot as plt
from nlp_utils.fileio import load_df_semantic
import nlp_utils as nu

from nlp_utils.corex_utils import calc_cov_corex, get_s_topic_words
# db_folder = r'/media/lee/Shared Storage'
db_folder = r'E:\\'

topic_model = cPickle.load(open(os.path.join('output', 'mod_cit_tree.pkl'), 'rb'))

s_anchor = get_s_topic_words(topic_model)

da_sigma, da_doc_topic = calc_cov_corex(topic_model, s_anchor.index, topic_model.docs.values)

#%%

con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

df_tm = load_df_semantic(con, topic_model.docs.values)
s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['year'], norm_each_topic=False)


from importlib import reload
reload(nu.plot)

highlight_topics = ['topic_' + str(i) for i in range(len(corex_anchors))]

year_range_fit = slice(2015,2020)
year_range_plot = slice(1990,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=30, highlight_topics=highlight_topics)



# %%

print('Generating Graph')

import networkx as nx 
G = nx.Graph() 

min_edge_weight = 1 #Minimum edge weight to be included.

for topic_i in da_sigma.coords['topic_i'].values:
    for topic_j in da_sigma.coords['topic_j'].values:
        weight = da_sigma.sel(topic_i=topic_i, topic_j=topic_j).item()
        if  weight > min_edge_weight:  
            if topic_i != topic_j:
                G.add_edge(topic_i,topic_j, weight=weight)


for node in G.nodes:
    G.nodes[node]['disp_text'] = s_anchor[node].replace(',', '\n')
    G.nodes[node]['size'] = np.sqrt(da_doc_topic.sel(topic=node).sum('doc').item()/10)


# %%

from community import community_louvain
import matplotlib.cm as cm
import networkx as nx

from bokeh.models.sources import ColumnDataSource, CustomJS
from bokeh.io import output_file, show, output_notebook
from bokeh.models import (Circle, HoverTool,
                          MultiLine, Range1d, TapTool, Div, Circle, MultiLine, Legend)
from bokeh.plotting import figure, show, from_networkx, save
from bokeh.palettes import Colorblind3, Colorblind4, Colorblind5, Colorblind6, Colorblind7
from bokeh.layouts import column, row, layout
from bokeh.palettes import Spectral3, Spectral4, Spectral5, Spectral6, Spectral7, Spectral
from bokeh.io import output_notebook, show


output_notebook()

partition = community_louvain.best_partition(G, resolution = 1)
num_partitions = len(set(partition.values()))

if num_partitions > 7:
    pal = Spectral
else:
    pal_dict = {1: Spectral3, 2: Spectral3, 3:Spectral3, 4 : Spectral4, 5 : Spectral5,  6: Spectral6, 7:Spectral7}
    pal = pal_dict[num_partitions]
fill_colors = [pal[i] for i in partition.values()]


def gen_graph_renderer(G, fill_colors, line_thickness, seed):
    """generate the renderer from the graph and metadata"""
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, seed=seed, center=(0, 0), k=0.05)

    graph_renderer.node_renderer.data_source.add(fill_colors, 'color')
    graph_renderer.node_renderer.data_source.add(line_thickness.values, 'line_thick')    

    graph_renderer.node_renderer.glyph = Circle(radius = 'size', fill_color = 'color', fill_alpha= 1, radius_units='screen', line_color = 'black', line_width='line_thick')
    graph_renderer.edge_renderer.glyph = MultiLine(line_width = 'weight', line_alpha = 0.3)

    return graph_renderer


graph_figure = figure(plot_width=900, plot_height=700,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("text", "@disp_text")])
node_tap_tool = TapTool()
graph_figure.add_tools(node_hover_tool, node_tap_tool)


graph_renderer = gen_graph_renderer(G, fill_colors, line_thickness=pd.Series(1,index=s_anchor.index.values), seed= 42)
graph_figure.renderers = [graph_renderer]

show(graph_figure)
# %%

# %%
