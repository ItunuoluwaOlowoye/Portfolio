import streamlit as st
from PIL import Image
import pydeck as pdk
import pandas as pd
import numpy as np
import udfs
import ast
import os

logo = Image.open('pictures/logo.png')
brand_name = 'Lotionfy'

st.set_page_config('Trade Area Modelling', page_icon=logo,
                   layout='wide')

with st.sidebar:
    st.header('App Overview')
    st.markdown(f'''<font style="font-size:14px;">We mined different web and map
                <font style="font-size:12px;">(Google Maps, Open Street Map, Open Cage Data)</font>
                sources to get data on the population in US neighborhoods,
                locations of department stores, the size and attractions around the stores
                <font style="font-size:12px;">(highways, commercial businesses, amenities, public transport system, walkways, open parking spaces etc)</font>,
                and the distance and travel time between neighborhoods and department stores.<br>
                We then created a customer base for a fictitious beauty and personal care brand in the US.</font>''',
                unsafe_allow_html=True)

logo_col, title_col = st.columns([1,9], gap='small')
logo_col.image(logo)
title_col.title('Trade Area Model')

case_study_tab, exploration_tab, tam_tab = st.tabs(['Case Study', 'Exploration', 'Trade Area'])

with case_study_tab:
    st.markdown(f'''{brand_name} is a fictitious beauty and personal care brand that has thrived as an exclusively
                online business post-pandemic, making more than a million dollars in sales per annum. They serve
                a predominantly US market, and have decided to build their physical presence.<br>
                To start, they have also decided to leverage partnerships with already existing 
                <a href="https://www.junglescout.com/wp-content/uploads/2023/09/Jungle-Scout-Consumer-Trends-Report-Q3-2023.pdf" style="color:#823DD3; text-decoration:None">top department stores</a>
                in the US. Walmart, Target, Kohl's, Marshalls and JCPenney have agreed to add the brand to their stock.<br>
                {brand_name} has then decided to use a **trade area model** to know which store location to begin their **physical footprint** from.
                The next two tabs show the model results.<br>
                <font style="font-size:14px;"> Click <a href="" style="color:#823DD3; text-decoration:None">put Github link</a>
                to see the data mining steps and <a href="" style="color:#823DD3; text-decoration:None">put Github link</a>
                to see the model source code.</font>''',
                unsafe_allow_html=True)

with exploration_tab:
    # read data
    customer_cities = udfs.load_data(file='data/customers/customer_cities.csv')
    customer_communities = udfs.load_data(file='data/customers/customer_communities.csv')
    
    # data transformation steps to get customer population in the cities and find the prime location
    customer_cities.index = customer_cities[['city','state']].apply(lambda x:', '.join(x), axis=1)
    customer_city_population = customer_communities.groupby(['location']).agg({'population':'sum'})
    customer_cities = pd.concat([customer_cities, customer_city_population], axis=1, ignore_index=False)\
        .sort_values('population', ascending=False).reset_index(names='location')
    prime_location = customer_cities.loc[0,'location']
    prime_location_lat = customer_cities.loc[0,'latitude']
    prime_location_lng = customer_cities.loc[0,'longitude']
    customer_cities['population_string'] = customer_cities['population'].apply(lambda x: '{:,}'.format(x))
    
    # create the map chart
    customer_cities_layer = pdk.Layer(type='ScatterplotLayer', data=customer_cities.drop(0, axis=0),
                                      get_position=['longitude','latitude'], pickable=True,
                                      get_color=[217, 185, 255, 160], get_radius='population/3',
                                      auto_highlight=True)
    customer_cities_prime_layer = pdk.Layer(type='ScatterplotLayer', data=customer_cities.loc[[0]],
                                      get_position=['longitude','latitude'], pickable=True,
                                      get_color=[130, 61, 211, 160], get_radius='population/3',
                                      auto_highlight=True)
    customer_cities_view_state = pdk.ViewState(longitude=-95.7128, latitude=37.0902,
                                               zoom=2.5, min_zoom=2, max_zoom=6, pitch=0)
    customer_cities_map = pdk.Deck(map_style=None, layers=[customer_cities_prime_layer, customer_cities_layer],
                                   initial_view_state=customer_cities_view_state,
                                   tooltip={"text": """City: {city}\nState: {state}\nPopulation: {population_string}"""}
                                   )
    
    # create bar chart of top ten cities
    customer_cities_bar = udfs.altair_bar_chart(df=customer_cities.head(10), xaxis='population', xtitle='Customers',
                                                textaxis='population_string', yaxis='city', tooltip=['city','state','population'],
                                                color='#D9B9FF')
    customer_cities_bar_highlight = udfs.bar_conditional_color(barchart=customer_cities_bar, words_list=[prime_location],
                                                               field='location', color='#823DD3')
    customer_cities_bar = customer_cities_bar + customer_cities_bar_highlight
    
    # visualize results
    st.markdown(f"""{brand_name}'s cityscape reveals a hotspot in Western United States, primarily
                     California. Their first physical footprint will be in the top customer city which is
                     <strong style="color:#823DD3;">{prime_location}.</strong> The next tab answers the
                     question of where in Los Angeles is best.""",
                     unsafe_allow_html=True)
    map_col, bar_col = st.columns([2,1], gap='small')
    with map_col.expander(f'Cityscape of {brand_name}: Unveiling Customer Hotspots', expanded=True):
        st.pydeck_chart(customer_cities_map, use_container_width=True)
    with bar_col.expander(f"City Spotlight: {brand_name}'s Top Customer Cities", expanded=True):
        st.altair_chart(customer_cities_bar, use_container_width=True)
    
with tam_tab:
    customer_communities['population_string'] = customer_communities['population'].apply(lambda x: '{:,}'.format(x))
    prime_community = customer_communities.loc[customer_communities['location'] == prime_location]
    st.write(prime_community.head())

    store_data = udfs.load_data('data/stores/store_data.csv')
    store_data['geometry'] = store_data['geometry'].apply(lambda x: udfs.wkt_polygon_to_coordinates(x))
    st.write(store_data.head(2))

    path_df = udfs.load_data('data/stores/path_to_store.csv')
    path_df = pd.melt(frame=path_df, id_vars='neighborhoods', var_name='store', value_name='path')
    path_df['path'] = path_df['path'].apply(ast.literal_eval)
    path_df['path'] = path_df['path'].apply(lambda x:[[coord[1],coord[0]] for coord in x])
    
    traffic_time = udfs.load_data('data/stores/traffic_time_mins.csv')
    traffic_time = pd.melt(frame=traffic_time, id_vars='neighborhoods', var_name='store', value_name='time')
    
    distance = udfs.load_data('data/stores/store_distances.csv')
    distance = pd.melt(frame=distance, id_vars='neighborhoods', var_name='store', value_name='distance')
    
    path_df = pd.merge(left=path_df, right=traffic_time, how='inner', on=['neighborhoods','store'])\
        .merge(right=distance, how='inner', on=['neighborhoods','store'])
    time_rank = path_df['time'].rank(pct=True, ascending=False)
    path_df.loc[time_rank<0.25, 'color'] = '#ED7D31'
    path_df.loc[(time_rank>=0.25) & (time_rank<0.5), 'color'] = '#F4B183'
    path_df.loc[(time_rank>=0.5) & (time_rank<0.75), 'color'] = '#8FAADC'
    path_df.loc[time_rank>=0.75, 'color'] = '#4472C4'
    path_df['color'] = path_df['color'].apply(udfs.hex_to_rgb)
    st.write(path_df.head())
    
    # create the map chart
    prime_community_layer = pdk.Layer(type='ScatterplotLayer', data=prime_community,
                                      get_position=['longitude','latitude'], pickable=True,
                                      get_color=[130, 61, 211, 160], get_radius='population/100',
                                      tooltip={"text": """Neighborhood: {neighborhoods}\nPopulation: {population_string}"""},
                                      auto_highlight=True)
    stores_layer = pdk.Layer(type='PolygonLayer', data=store_data, get_polygon='geometry', filled=True,
                             get_fill_color=[130, 61, 211, 160], get_line_color=[130, 61, 211, 160], get_line_width=100,
                             tooltip={"text": """Store: {name}\nAddress: {address}"""},
                             pickable=True, auto_highlight=True)
    path_layer = pdk.Layer(type='PathLayer', data=path_df, pickable=True, get_color='color', width_scale=20,
                           width_min_pixels=2, get_path='path', get_width=5, auto_highlight=True,
                           tooltip={'text': """From: {neighborhoods}\nTo: {store}"""})
    prime_community_view_state = pdk.ViewState(longitude=prime_location_lng, latitude=prime_location_lat,
                                               zoom=10, min_zoom=2, pitch=50)
    prime_community_map = pdk.Deck(map_style=None, layers=[prime_community_layer, stores_layer, path_layer],
                                   initial_view_state=prime_community_view_state,
                                   )
    st.pydeck_chart(prime_community_map, use_container_width=True)