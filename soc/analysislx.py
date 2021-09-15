#%%

import pandas as pd
import os
import sqlite3
import networkx as nx

from nlp_utils.fileio import load_df_semantic

db_folder = r'/media/lee/Shared Storage'

con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

# cursor.execute("SELECT id from raw_text ORDER BY RANDOM() LIMIT 1000")
cursor.execute("SELECT id from raw_text LIMIT 1000")
results = cursor.fetchall()

ids = [t[0] for t in results]


df = load_df_semantic(con, ids)


# %%

df_1 = df.where(df['paperAbstract'].str.contains('energy storage')).dropna(how='all')
df_1 = df_1.iloc[3].to_frame().T

# df_1 = df.iloc[5].to_frame().T
print(df_1['title'].values)
#%%


from nlp_utils.citation import gen_citation_graph, trim_graph

G = gen_citation_graph(df_1)

df_2 = load_df_semantic(con, G.nodes)

G = gen_citation_graph(df_2)
G = trim_graph(G, con, frac_keep_factor=5, min_connect=2)

G.number_of_nodes()

for node in G.nodes:
    title = df_2.loc[node]['title']
    G.nodes[node]['title']= title


#%%
G = nx.Graph(G)
G.remove_node(df_1.index[0])

#%%
G

s_degrees = pd.Series(dict(G.degree()))
s_degrees.hist(bins=100)
# df.loc[list(G.nodes)[0]
#%%
from bokeh.plotting import figure, from_networkx
from bokeh.io import output_notebook, show, output_file
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,)
from bokeh.palettes import Spectral4

plot = figure(plot_width=1000, plot_height=800)
graph = from_networkx(G, nx.spring_layout)

graph.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
graph.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
graph.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

graph.edge_renderer.glyph = MultiLine(line_color='black', line_alpha=0.4, line_width=5)
graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)


graph.selection_policy = NodesAndLinkedEdges()
# graph.inspection_policy = EdgesAndLinkedNodes()



plot.add_tools(HoverTool(tooltips=[('title', '@title')]), TapTool(), BoxSelectTool())


plot.renderers.append(graph)

output_notebook()
show(plot)

# output_file('test.html')
# show(plot)

#%%

# from networkx_viewer import Viewer

# app = Viewer(G)
# app.mainloop()



#%%


con.close()