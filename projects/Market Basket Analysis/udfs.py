# import packages
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from pyvis.network import Network
from pyvis.options import Layout
from graphviz import Digraph
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np

frequent_itemset = 30
random_seed = 42

# define function to convert from binary to boolean
def encoding(x):
    if x<=0: return False
    else: return True

# analyze in store case study
@st.cache_data(ttl=12*60*60)
def instore_case_study(file:str='projects/Market Basket Analysis/store_data.csv'):
    # load data
    orders = pd.read_csv(file, header=None, sep='\t')
    # each row represents a unique order
    # set the column name
    orders.columns = ['product']
    # create a column with the index to represent the order number
    orders['id'] = orders.index + 1
    # create a list from the products listed in each row
    orders['product'] = orders['product'].str.split(',')
    # put each order product in its own row
    orders = orders.explode('product', ignore_index=True)
    # remove null or empty products from the data
    orders['product'].dropna(inplace=True)
    orders = orders.loc[orders['product'] != '', :]
    # strip any whitespace
    orders['product'] = orders['product'].str.strip()
    # quantify the items in the orders
    orders = orders.groupby(['id','product'], as_index=False).size()
    # transform orders data to one-hot encoded format
    market_basket = pd.crosstab(orders['id'], orders['product'])
    # convert from binary to boolean
    market_basket = market_basket.map(encoding)
    # calculate minimum support threshold
    frequent_itemset = 30
    total_purchases = market_basket.index.nunique()
    min_support = frequent_itemset/total_purchases
    frequent_items = apriori(market_basket, min_support=min_support, max_len=2, use_colnames=True)
    # define association rules and unpack consequents
    rules = association_rules(frequent_items, metric='lift', min_threshold=1).sort_values('confidence', ascending=False)
    rules = rules.explode('antecedents').explode('consequents')
    # set higher confidence itemsets to be itemsets with confidence greater than the 75th percentile confidence value
    shopping_exp_df = rules.loc[rules['confidence'] > rules['confidence'].mean(),
                                ['antecedents','consequents','support','confidence','lift']]
    shopping_exp_df['title'] = shopping_exp_df\
        .agg(lambda x: f"{round(x['confidence']*100, 1)}% of customers who buy {x['antecedents']} also buy {x['consequents']}\nCustomers are {round(x['lift'], 2)} times more likely to buy {x['consequents']} if they also buy {x['antecedents']}",
             axis=1)
    # calculate the quantity of products sold
    items_move = orders.groupby('product').agg({'size':'sum'})
    # set slow moving items as products whose sold units are less than the average quantity of products sold
    slower_moving_items = items_move[items_move['size']<= items_move['size'].mean()].index.unique()
    # filter the association results to associations whose consequents are part of the slow moving items
    slower_moving_items_df = rules.loc[(rules['consequents'].isin(slower_moving_items))]
    return shopping_exp_df, slower_moving_items, slower_moving_items_df

# analyze online case study
@st.cache_data(ttl=12*60*60)
def online_case_study(file:str='projects/Market Basket Analysis/MarketingData/Online_Sales.csv'):
    # load data
    orders = pd.read_csv(file)
    # transform orders data to one-hot encoded format
    market_basket = pd.crosstab(orders['Transaction_ID'], orders['Product_Description'])
    # convert from binary to boolean
    market_basket = market_basket.map(encoding)
    # calculate minimum support threshold
    frequent_itemset = 30
    total_purchases = orders['Transaction_ID'].nunique()
    min_support = frequent_itemset/total_purchases
    # create itemsets
    frequent_items = apriori(market_basket, min_support=min_support, use_colnames=True)
    # define association rules and unpack consequents
    rules = association_rules(frequent_items, metric='lift', min_threshold=1).sort_values('confidence', ascending=False)
    rules = rules.explode('consequents')
    # set higher confidence itemsets to be itemsets with confidence greater than the mean confidence value
    cross_selling_exp_df = rules.loc[rules['confidence'] > rules['confidence'].mean(), :]
    # create a price list for all products using the most recent prices
    price_list = orders.groupby('Product_Description', as_index=False).agg({'Avg_Price':'last'})
    # find prices of antecedents and consequents
    antecedent_prices = cross_selling_exp_df['antecedents'].apply(lambda x: [price_list.loc[price_list['Product_Description'] == elem,
                                                                                'Avg_Price'].iloc[0] for elem in x])
    consequent_prices = cross_selling_exp_df['consequents'].apply(lambda x: [price_list.loc[price_list['Product_Description'] == x,
                                                                                'Avg_Price'].iloc[0]])
    total_prices = antecedent_prices + consequent_prices
    # price of package deal
    cross_selling_exp_df.loc[:, 'price'] = total_prices.apply(lambda x: sum(x))
    # set higher revenue itemsets to be itemsets with price greater than the mean price
    higher_rev_pkg_deals = cross_selling_exp_df.loc[cross_selling_exp_df['price'] > cross_selling_exp_df['price'].mean(), :]\
        .sort_values('price', ascending=False)
    return cross_selling_exp_df, higher_rev_pkg_deals

# draw network graph for associations
@st.cache_data(ttl=1*60*60)
def assc_graph(df, src, dest, nodes_list, nodes_list_color, other_color):
    # create network graph
    G = nx.from_pandas_edgelist(df=df, source=src, target=dest, edge_attr=True, create_using=nx.MultiDiGraph)
    # separate antecedents and consequents in a new network graph by color
    antecedents = []
    antecedents.append(np.setdiff1d(G.nodes(),nodes_list,assume_unique=False))
    G_viz=nx.MultiDiGraph()
    G_viz.add_nodes_from(antecedents[0], color=other_color)
    G_viz.add_nodes_from(nodes_list, color=nodes_list_color, size=16)
    G_viz.add_edges_from(G.edges(data=True))
        
    # build the pyvis visuals
    layout_net = Layout(randomSeed=random_seed)
    net = Network(layout=layout_net, directed=True, bgcolor='#0E1117', font_color='#FAFAFA', notebook=False,
                  cdn_resources='remote')
    # make the graph as static as possible
    net.repulsion()
            
    # draw the graph
    net.from_nx(G_viz,show_edge_weights=True)
    net.set_options('''
                    const options = {"nodes": {"borderWidth": 2, "borderWidthSelected": 10,
                    "font": {"face": "verdana"}, "shapeProperties": {"borderRadius": 20}},
                    "edges": {"font": {"face": "verdana"}, "selectionWidth": 10}, 
                    "interaction": {"hover": true}}
                    ''')
    html_chart = net.generate_html()
    return html_chart

