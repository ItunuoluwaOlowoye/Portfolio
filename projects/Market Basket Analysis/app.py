# import packages
import streamlit as st
import pandas as pd
import udfs

# set default variables
random_seed = 42
primary_color = '#8e43e7'

# set page layout
st.set_page_config(page_title='Market Basket Analysis', page_icon='🛒', layout='wide')

# set page title
st.title('🛒 Market Basket Analysis')

# introduction
placeholder = st.empty()
with placeholder.container():
    st.markdown(f"""Market Basket Analysis (MBA) is used in the retail industry to know what products customers are
                buying together. Retailers can leverage this to <strong style="color:{primary_color};">cross-sell 
                products, improve customer experience, increase impulse purchases made by customers, and offer
                promotions on associated products.</strong><br>
                For in-store retailers, this means optimizing the store layout for product placement and display.
                For example, when customers buy spaghetti, they may also buy tomato soup cans with it so the store
                can display spaghetti and tomato soup cans close to each other.<br>
                For online retailers, this means personalizing suggestions to customers in the *"These products
                may also interest you", "Customers who bought this also bought...", etc.* categories. For example,
                customers who bought an iPhone 15 also bought a Lightning USB-C to iPhone charger.""",
                unsafe_allow_html=True)
    instore_study = st.button('In-store Case Study', type='primary')
    online_study = st.button('Online Case Study', type='primary')

# instore case study
if instore_study:
    st.session_state['type'] = 'instore'
elif online_study:
    st.session_state['type'] = 'online'

if 'type' not in st.session_state: st.stop()

elif st.session_state.type == 'instore':
    # run model
    shopping_exp_df, slower_moving_items, slower_moving_items_df = udfs.instore_case_study()
    # clear introduction and add a back button
    placeholder.empty()
    back_button = st.button('◀️ Back')
    if back_button: st.session_state.clear(); st.rerun()
    # create tabs for summary and model results
    case_study, shopping_exp, slow_items = st.tabs(['Case study', 'Enhancing shopping experiences',
                                                    'Addressing slow moving items'])
    # case study summary
    with case_study:
        st.markdown(f"""Emily is the new store manager at a physical grocery store (we'll call them SunnyVille). In her
                    first couple of months familiarizing herself with the role, she couldn't help but notice a slight 
                    disconnect. Products that naturally complemented each other were scattered across the store. A 
                    customer buying eggs, for example, had to walk to the other end of the store, if they wanted to 
                    include milk. She also noticed from the inventory that there were some products that just didn't sell
                    out fast enough.<br><br>
                    From these, she discovered that there was potential to enhance shopping experiences for their
                    customers by transforming the store layout to create "mini-environments" where related items
                    harmonised with each other. Strategically placed slow moving items with promotions could also find
                    their way into customers' baskets and reduce the inventory stock.""",
                    unsafe_allow_html=True)
        st.link_button(label='Model source code 🔗', url='')
    
    # enhancing shopping experiences
    with shopping_exp:
        # create list of product options
        antecedents = shopping_exp_df['antecedents'].tolist()
        consequents = shopping_exp_df['consequents'].tolist()
        product_options = list(set(antecedents + consequents))
        # create columns for filter and model results
        filter_col, model_col = st.columns([1,4], gap='small')
        # create multiselection for filter
        selected_items = filter_col.multiselect('Choose product(s)', options=product_options, label_visibility='collapsed',
                                                 placeholder='Select the product(s) to view')
        if selected_items == []: selected_items = product_options
        # filter dataframe to selected products
        shopping_exp_df = shopping_exp_df.loc[(shopping_exp_df['antecedents'].isin(selected_items)) |
                                               shopping_exp_df['consequents'].isin(selected_items), :]
        consequents = shopping_exp_df['consequents'].tolist()
        # build associated graphs
        pos_graph = udfs.assc_graph(df=shopping_exp_df, src='antecedents', dest='consequents', nodes_list=consequents,
                                    nodes_list_color=primary_color, other_color='grey')
        with model_col:
            with st.spinner('Loading graph...'):
                st.components.v1.html(pos_graph, height=600)
