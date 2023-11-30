from shapely import Polygon, wkt
import geopandas as gpd
import streamlit as st
from PIL import Image
import pydeck as pdk
import pandas as pd
import numpy as np
import udfs
import ast
import os

# set logo and brand name
logo = Image.open('pictures/logo.png')
brand_name = 'Lotionfy'

# set colors
primary_color = '#8e43e7'
primary_background_color = '#e3d0f9'
primary_store_color = '#17a2b8'
secondary_store_color = '#b9e3ea'
features_color = '#273c66'
vgood_traffic = '#4472c4'
good_traffic = '#8faadc'
bad_traffic = '#f4b183'
vbad_traffic = '#ed7d31'

icon_properties = {"url": "", "width": 242, "height": 242, "anchorY": 242}
store_icon = r'https://upload.wikimedia.org/wikipedia/commons/5/58/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Department_Store_%E2%80%93_Stores_%E2%80%93_white.png'
business_icon = 'https://upload.wikimedia.org/wikipedia/commons/7/78/Concrete_Jungle_Icon.png'
best_store_icon = r'https://upload.wikimedia.org/wikipedia/commons/c/ca/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Department_Store_%E2%80%93_Stores_%E2%80%93_default.png'
busstop_icon = r'https://upload.wikimedia.org/wikipedia/commons/6/63/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Bus_Stop_%E2%80%93_Transportation_%E2%80%93_Dark.png'
parking_icon = r'https://upload.wikimedia.org/wikipedia/commons/a/a7/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Subway_%E2%80%93_Transportation_%E2%80%93_Default.png'
subway_icon = r'https://upload.wikimedia.org/wikipedia/commons/3/34/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Parking_%E2%80%93_Transportation_%E2%80%93_Dark.png'

st.set_page_config('Trade Area Modelling', page_icon=logo, menu_items={"About": "# About"},
                   layout='wide')

with st.sidebar:
    st.header('App Overview')
    st.markdown(f'''<font style="font-size:14px;">We mined different web <font style="font-size:12px;">(StatisticalAtlas, Zillow)</font>
                and map <font style="font-size:12px;">(Google Maps, Open Street Map, Open Cage Data)</font>
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
                <a href="https://www.junglescout.com/wp-content/uploads/2023/09/Jungle-Scout-Consumer-Trends-Report-Q3-2023.pdf" style="color:{primary_color}; text-decoration:None">top department stores</a>
                in the US. Walmart, Target, Kohl's, Marshalls and JCPenney have agreed to add the brand to their stock.<br>
                {brand_name} has then decided to use a **trade area model** to know which store location to begin their **physical footprint** from.
                The next two tabs show the model results.<br>
                <font style="font-size:14px;"> Click <a href="" style="color:{primary_color}; text-decoration:None">put Github link</a>
                to see the data mining steps and <a href="" style="color:{primary_color}; text-decoration:None">put Github link</a>
                to see the model source code.</font>''',
                unsafe_allow_html=True)

with exploration_tab:
    # read data
    customer_cities = udfs.load_data(file='data/customers/customer_cities.csv')
    customer_communities = udfs.load_data(file='data/customers/customer_communities.csv')
    
    # data transformation steps to get customer population in the cities and find the prime location
    customer_communities = customer_communities.loc[customer_communities['population']>0, :]
    customer_cities.index = customer_cities[['city','state']].apply(lambda x:', '.join(x), axis=1)
    customer_city_population = customer_communities.groupby(['location']).agg({'population':'sum'})
    customer_cities = pd.concat([customer_cities, customer_city_population], axis=1, ignore_index=False)\
        .sort_values('population', ascending=False).reset_index(names='location')
    customer_cities = customer_cities.loc[customer_cities['population'].notna()]
    prime_location = customer_cities.loc[0,'location']
    prime_location_lat = customer_cities.loc[0,'latitude']
    prime_location_lng = customer_cities.loc[0,'longitude']
    customer_cities['population_string'] = customer_cities['population'].astype(int).apply(lambda x: '{:,}'.format(x))
    
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
                                                color=primary_background_color)
    customer_cities_bar_highlight = udfs.bar_conditional_color(barchart=customer_cities_bar, words_list=[prime_location],
                                                               field='location', color=primary_color)
    customer_cities_bar = customer_cities_bar + customer_cities_bar_highlight
    
    # visualize results
    st.markdown(f"""{brand_name}'s cityscape reveals a hotspot in Western United States, primarily
                     California. Their first physical footprint will be in the top customer city which is
                     <strong style="color:{primary_color};">{prime_location}.</strong> The next tab answers the
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
    prime_community['geometry'] = prime_community['geometry'].apply(lambda x: udfs.wkt_polygon_to_coordinates(x))
    
    stores = udfs.load_data('data/stores/store_data.csv')
    store_data = stores.copy()
    stores['geometry'] = stores['geometry'].apply(lambda x: udfs.wkt_polygon_to_coordinates(x))
    store_data['geometry'] = store_data['geometry'].apply(wkt.loads)
    store_data['lng'] = store_data['geometry'].apply(lambda x: x.centroid.x)
    store_data['lat'] = store_data['geometry'].apply(lambda x: x.centroid.y)
    store_data.drop('geometry',axis=1,inplace=True)
    store_icon_data = icon_properties.copy()
    store_icon_data['url'] = store_icon
    store_data["store_icon_data"] = None
    for i in store_data.index:
        store_data["store_icon_data"][i] = store_icon_data
    
    store_distances = udfs.load_data('data/stores/store_distances.csv')
    store_preferences = udfs.load_data('data/stores/store_preferences.csv')

    # filter for population in communities with store distances
    community_pop = customer_communities.loc[(customer_communities['location'].isin(store_data['city_location'])),
                                             ['neighborhoods','population']]
    
    # set neighborhoods as index in community expense and store distances
    community_pop.set_index('neighborhoods', inplace=True)
    store_distances.set_index('neighborhoods', inplace=True)

    # set store as index
    store_preferences.set_index('store', inplace=True)

    # rank travel time in descending order
    travel_time_rank = store_preferences[['avg_travel_time']].rank(pct=True, ascending=False)
    # rank other store preferences
    store_preferences_rank = store_preferences.drop(['avg_travel_time'], axis=1).rank(pct=True)
    # create ranked dataframe
    store_preferences_rank = pd.concat([travel_time_rank,store_preferences_rank], axis=1)

    # measure attractiveness
    attractiveness = store_preferences_rank.sum(axis=1)

    # create the dataframe that serves as the numerator in the Huff model
    numerator_df = pd.DataFrame([], index=store_distances.index)

    # calculate the numerator
    for col in store_distances.columns:
        numerator_df[col] = attractiveness[col] / (store_distances[col])**2

    # calculate the denominator
    denominator = numerator_df.sum(axis=1)

    # create the probability dataframe
    prob_df = pd.DataFrame([], index=store_distances.index)

    for col in numerator_df.columns:
        prob_df[col] = numerator_df[col]/denominator

    # create the population dataframe
    pop_df = pd.DataFrame([], index=store_distances.index)

    # calculate the probable number of store walk-ins in each neighborhood
    for col in prob_df.columns:
        pop_df[col] = prob_df[col] * community_pop['population']

    # sum up the total probable customer walk-ins at each store location
    population = pop_df.sum(axis=0).astype(int).to_dict()

    # select the store with the max probable sales
    max_population = max(population.values())
    optimum_stores = [key for key in population.keys() if population[key] == max_population]
    # print result
    if len(optimum_stores) == 1:
        prime_store_location = f'''The optimum store location is {optimum_stores[0]} with probable walk-ins of {max_population:,} existing customers'''
    else:
        optimum_stores = "\n".join(optimum_stores)
        prime_store_location = f'''The optimum store locations with probable walk-ins of {max_population:,} existing customers are:\n\n{optimum_stores}'''
    
    prime_store = stores.loc[stores['store_location'].isin(optimum_stores), 'geometry'].iloc[0]
    prime_store_geom = (Polygon(prime_store[0]).centroid.x, Polygon(prime_store[0]).centroid.y)
    prime_store_lng, prime_store_lat = prime_store_geom[0], prime_store_geom[1]

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
    path_df.loc[time_rank<0.25, 'color'] = vbad_traffic
    path_df.loc[(time_rank>=0.25) & (time_rank<0.5), 'color'] = bad_traffic
    path_df.loc[(time_rank>=0.5) & (time_rank<0.75), 'color'] = good_traffic
    path_df.loc[time_rank>=0.75, 'color'] = vgood_traffic
    path_df['color'] = path_df['color'].apply(udfs.hex_to_rgb)
    
    business_comm = udfs.load_data('data/stores/business_communities.csv')
    business_comm = business_comm.loc[business_comm['name'].notna(), :]
    business_comm['geometry'] = business_comm['geometry'].apply(wkt.loads)
    business_comm['lng'] = business_comm['geometry'].apply(lambda x: x.centroid.x)
    business_comm['lat'] = business_comm['geometry'].apply(lambda x: x.centroid.y)
    business_comm.drop('geometry',axis=1, inplace=True)
    business_icon_data = icon_properties.copy()
    business_icon_data['url'] = business_icon
    business_comm["business_icon_data"] = None
    for i in business_comm.index:
        business_comm["business_icon_data"][i] = business_icon_data
    
    highways = udfs.load_data('data/stores/highways.csv')
    highways['geometry'] = highways['geometry'].apply(udfs.wkt_polygon_to_coordinates)
    #highways['geometry'] = highways['geometry'].apply(lambda x:x[0])
    st.write(path_df.head())
    st.write(highways.head())

    # create the map chart
    alpha = [160]
    prime_community_layer = pdk.Layer(type='PolygonLayer', data=prime_community,
                                      get_polygon='geometry', pickable=True, filled=False,
                                      get_line_color=list(udfs.hex_to_rgb(primary_color)),
                                      get_line_width=50, auto_highlight=True,
                                      tooltip={"text": """Neighborhood: {neighborhoods}\nPopulation: {population_string}"""}
                                      )
    stores_poly_layer = pdk.Layer(type='PolygonLayer', data=stores, get_polygon='geometry', filled=True,
                             get_fill_color=list(udfs.hex_to_rgb(primary_store_color)) + alpha,
                             get_line_color=list(udfs.hex_to_rgb(primary_store_color)),
                             get_line_width=10, pickable=True, auto_highlight=True, extruded=False,
                             tooltip={"text": """Store: {name}\nAddress: {address}"""}
                             )
    stores_layer = pdk.Layer(type="IconLayer", data=store_data, get_icon="store_icon_data", get_size=2, size_scale=15,
                             get_position=["lng", "lat"], pickable=True,
                             tooltip={"text": """Store: {name}\nAddress: {address}"""}
                             )
    dt_layer = pdk.Layer(type='PathLayer', data=path_df, pickable=True, get_color='color', width_scale=10,
                           width_min_pixels=2, get_path='path', get_width=1, auto_highlight=True,
                           tooltip={'text': """From: {neighborhoods}\nTo: {store}\nTime: {time} mins\nDistance: {distance} km"""})
    business_comm_layer = pdk.Layer(type="IconLayer", data=business_comm, get_icon="business_icon_data", get_size=1.5,
                                    size_scale=15, get_position=["lng", "lat"], pickable=True,
                                    tooltip={'text': """Name: {name}"""})
    highways_layer = pdk.Layer(type='PathLayer', data=highways, pickable=True, get_color=list(udfs.hex_to_rgb(vgood_traffic)),
                               width_scale=10,
                           width_min_pixels=2, get_path='geometry', get_width=1, auto_highlight=True,
                           tooltip={'text': """Name: {name}"""})
    prime_community_view_state = pdk.ViewState(longitude=prime_store_lng, latitude=prime_store_lat,
                                               zoom=10, min_zoom=2, pitch=50)
    prime_community_map = pdk.Deck(map_style=None, #map_style="mapbox://styles/mapbox/dark-v11",
                                   layers=[stores_poly_layer, stores_layer, business_comm_layer, highways_layer],
                                   initial_view_state=prime_community_view_state,
                                   tooltip={"text": """Neighborhood: {neighborhoods}\nStore: {name}\nTime: {time}\nDistance: {distance}"""},
                                    )
    st.pydeck_chart(prime_community_map, use_container_width=True)