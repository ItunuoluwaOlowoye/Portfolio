from shapely import Polygon
import streamlit as st
from PIL import Image
import pydeck as pdk
import pandas as pd
import udfs
import ast

# set logo and brand name
logo = Image.open('pictures/logo.png')
brand_name = 'Lotionfy'

# set colors
primary_color = '#8e43e7'
primary_background_color = '#e3d0f9'
features_color = '#17a2b8'
vgood_traffic = '#4472c4'
good_traffic = '#8faadc'
bad_traffic = '#f4b183'
vbad_traffic = '#ed7d31'
good_choice = '#28a745'
fair_choice = '#ffc107'
bad_choice = '#dc3545'

# set icons
icon_properties = {"url": "", "width": 242, "height": 242, "anchorY": 242}
store_icon_url = r'https://upload.wikimedia.org/wikipedia/commons/c/ca/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Department_Store_%E2%80%93_Stores_%E2%80%93_default.png'
business_icon_url = 'https://upload.wikimedia.org/wikipedia/commons/7/78/Concrete_Jungle_Icon.png'
best_store_icon_url = r'https://upload.wikimedia.org/wikipedia/commons/e/e4/Solid_Bright_Green_Star_1.png'
busstop_icon_url = r'https://upload.wikimedia.org/wikipedia/commons/6/63/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Bus_Stop_%E2%80%93_Transportation_%E2%80%93_Dark.png'
parking_icon_url = r'https://upload.wikimedia.org/wikipedia/commons/a/a7/Map_marker_icon_%E2%80%93_Nicolas_Mollet_%E2%80%93_Subway_%E2%80%93_Transportation_%E2%80%93_Default.png'

# set page configuration
st.set_page_config('Trade Area Modelling', page_icon=logo, layout='wide', initial_sidebar_state='collapsed')

# configure sidebar
with st.sidebar:
    st.header('App Overview')
    st.markdown(f'''<font style="font-size:14px;">We mined different web <font style="font-size:12px;">(StatisticalAtlas, Zillow)</font>
                and map <font style="font-size:12px;">(Google Maps, Open Street Map, Open Cage Data)</font>
                sources to get data about neighborhoods in the US,
                locations of department stores, the size and attractions around the stores
                <font style="font-size:12px;">(highways, commercial businesses, amenities, public transport system, walkways, open parking spaces etc)</font>,
                and the distance and travel time between neighborhoods and department stores.<br></font>''',
                unsafe_allow_html=True)

# set page header
logo_col, title_col = st.columns([1,9], gap='small')
logo_col.image(logo)
title_col.title('Trade Area Model')

# DATA ANALYSES
# read data
customer_cities = udfs.load_data(file='data/customers/customer_cities.csv')
customer_communities = udfs.load_data(file='data/customers/customer_communities.csv')
store_data = udfs.load_data('data/stores/store_data.csv')
store_distances = udfs.load_data('data/stores/store_distances.csv')
store_preferences = udfs.load_data('data/stores/store_preferences.csv')
business_comm = udfs.load_data('data/stores/business_communities.csv')
busstop = udfs.load_data('data/stores/roads.csv')
parking = udfs.load_data('data/stores/parking_spaces.csv')
highways = udfs.load_data('data/stores/highways.csv')
path_df = udfs.load_data('data/stores/path_to_store.csv')
traffic_time = udfs.load_data('data/stores/traffic_time_mins.csv')
distance = udfs.load_data('data/stores/store_distances.csv')

# data transformation steps to get customer population in the cities
customer_communities = customer_communities.loc[customer_communities['population']>0, :]
customer_communities.loc[:,'population_string'] = customer_communities['population'].apply(lambda x: '{:,}'.format(x))
customer_cities.index = customer_cities[['city','state']].apply(lambda x:', '.join(x), axis=1)
customer_city_population = customer_communities.groupby(['location']).agg({'population':'sum'})
customer_cities = pd.concat([customer_cities, customer_city_population], axis=1, ignore_index=False)\
    .sort_values('population', ascending=False).reset_index(names='location')
customer_cities.loc[:, 'population_string'] = customer_cities['population'].apply(lambda x: '{:,}'.format(x))

# find the prime city (city with most existing customers)
prime_city = customer_cities.loc[0,'location']
prime_city_lat = customer_cities.loc[0,'latitude']
prime_city_lng = customer_cities.loc[0,'longitude']

# get communities within the prime city
prime_community = customer_communities.loc[customer_communities['location'] == prime_city]
prime_community.loc[:, 'geometry'] = prime_community['geometry'].apply(lambda x: udfs.wkt_to_coordinates(x))

# build trade area model
pop_df, population, max_population, optimum_store = udfs.trade_area_model(customer_communities=customer_communities,
                                                                             store_data=store_data, store_distances=store_distances,
                                                                             store_preferences=store_preferences)

# add store icon to store data
store_data.loc[:, 'geopolygon'] = store_data['geometry'].apply(lambda x: udfs.wkt_to_coordinates(x))

# select prime stores and their coordinates
prime_store_data = store_data.loc[store_data['store_location'] == optimum_store, :].reset_index()
prime_store_data = udfs.add_icon_to_df(df=prime_store_data, icon_properties=icon_properties, icon_url=best_store_icon_url)
prime_store_data['lng'] = prime_store_data['geopolygon'].apply(lambda x: Polygon(x[0]).centroid.x)
prime_store_data['lat'] = prime_store_data['geopolygon'].apply(lambda x: Polygon(x[0]).centroid.y)
prime_store_lng, prime_store_lat = prime_store_data.loc[0, 'lng'], prime_store_data.loc[0, 'lat']

# other stores; merge together
other_stores_data = store_data.loc[store_data['store_location'] != optimum_store, :].reset_index()
other_stores_data = udfs.add_icon_to_df(df=other_stores_data, icon_properties=icon_properties, icon_url=store_icon_url)
prepped_store_data = pd.concat([prime_store_data, other_stores_data])

# set data for commercial buildings
business_comm = business_comm.loc[business_comm['name'].notna(), :]
business_comm = udfs.add_icon_to_df(df=business_comm, icon_properties=icon_properties, icon_url=business_icon_url)

# set data for busstops
busstop.loc[:, ['name','alt_name']].fillna('Not provided', inplace=True)
busstop = busstop.loc[busstop['highway'] == 'bus_stop', :]
busstop = udfs.add_icon_to_df(df=busstop, icon_properties=icon_properties, icon_url=busstop_icon_url)

# set data for parking
parking['name'].fillna('Not provided', inplace=True)
parking = udfs.add_icon_to_df(df=parking, icon_properties=icon_properties, icon_url=parking_icon_url)

# set highways
highways['name'].fillna('Not provided',inplace=True)
highways['geometry'] = highways['geometry'].apply(udfs.wkt_to_coordinates)

# set path data
path_df = pd.melt(frame=path_df, id_vars='neighborhoods', var_name='store', value_name='path')
path_df['path'] = path_df['path'].apply(ast.literal_eval).apply(lambda x:[[coord[1],coord[0]] for coord in x])
    
traffic_time = pd.melt(frame=traffic_time, id_vars='neighborhoods', var_name='store', value_name='time')
distance = pd.melt(frame=distance, id_vars='neighborhoods', var_name='store', value_name='distance')
    
path_df = pd.merge(left=path_df, right=traffic_time, how='inner', on=['neighborhoods','store'])\
    .merge(right=distance, how='inner', on=['neighborhoods','store'])
time_rank = path_df['time'].rank(pct=True, ascending=False)
path_df.loc[time_rank<0.25, 'color'] = vbad_traffic
path_df.loc[(time_rank>=0.25) & (time_rank<0.5), 'color'] = bad_traffic
path_df.loc[(time_rank>=0.5) & (time_rank<0.75), 'color'] = good_traffic
path_df.loc[time_rank>=0.75, 'color'] = vgood_traffic
path_df['color'] = path_df['color'].apply(udfs.hex_to_rgb)

# STORE STATISTICS
# create store locations list
store_locations = prepped_store_data['store_location'].unique()

# create store stats from distances, preferences, and predicted walk-in population
avg_distance = round(store_distances.mean(numeric_only=True), 1).to_frame(name='avg_distance')
store_stats = pd.concat([store_preferences, avg_distance], axis=1, ignore_index=False)
store_stats['population'] = store_stats.index.map(population)

# create a column with the choice rank
store_stats['choice'] = ''
store_stats.loc[store_stats['population'] > store_stats['population'].quantile(0.75), 'choice'] = 'good'
store_stats.loc[store_stats['population'] < store_stats['population'].quantile(0.5), 'choice'] = 'bad'
store_stats.loc[store_stats['population'] == store_stats['population'].max(), 'choice'] = 'best'
store_stats.loc[store_stats['choice'] == '', 'choice'] = 'fair'

# rank design and accessibility
store_stats.loc[:, ['design','accessibility']] = round(store_stats.loc[:, ['design','accessibility']].rank(pct=True) * 100, 2)


# DATA VIZ
# create visualization tabs
case_study_tab, exploration_tab, tam_tab = st.tabs(['Case Study', 'Exploration', 'Trade Area'])

# case study background
with case_study_tab:
    st.markdown(f'''A particular beauty and personal care brand that has thrived as an exclusively
                online business post-pandemic (we'll call them {brand_name}). They serve
                a predominantly US market, and have decided to build their physical presence.<br>
                To start, they have also decided to leverage partnerships with already existing 
                <a href="https://www.junglescout.com/wp-content/uploads/2023/09/Jungle-Scout-Consumer-Trends-Report-Q3-2023.pdf" style="color:{primary_color}; text-decoration:None">top department stores</a>
                in the US. Walmart, Target, Kohl's, Marshalls and JCPenney have agreed to add the brand to their stock.<br>
                {brand_name} has then decided to use a **trade area model** to know which store location to begin their **physical footprint** from.
                The *Exploration* and *Trade Area* tabs show the model results.<br>
                <font style="font-size:14px;"> Click <a href="" style="color:{primary_color}; text-decoration:None">put Github link</a>
                to see the data mining steps and <a href="" style="color:{primary_color}; text-decoration:None">put Github link</a>
                to see the model source code.</font>''',
                unsafe_allow_html=True)

# initial exploration
with exploration_tab:
    # create the map chart
    customer_cities_layer = pdk.Layer(type='ScatterplotLayer', data=customer_cities.drop(0, axis=0),
                                      get_position=['longitude','latitude'], pickable=True,
                                      get_color=[217, 185, 255, 160], get_radius='population/3',
                                      auto_highlight=True)
    customer_cities_prime_layer = pdk.Layer(type='ScatterplotLayer', data=customer_cities.loc[[0]],
                                      get_position=['longitude','latitude'], pickable=True,
                                      get_color=[130, 61, 211, 160], get_radius='population/3',
                                      auto_highlight=True)
    # US center as initial view state
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
    customer_cities_bar_highlight = udfs.bar_conditional_color(barchart=customer_cities_bar, words_list=[prime_city],
                                                               field='location', color=primary_color)
    customer_cities_bar = customer_cities_bar + customer_cities_bar_highlight
    
    # visualize results
    st.markdown(f"""{brand_name}'s cityscape reveals a hotspot in Western United States.
                 Their first physical footprint will be in the top customer city which is
                 <strong style="color:{primary_color};">{prime_city}.</strong><br>**Which department store should
                 {brand_name} start from?** The *Trade Area* tab answers this question.""",
                 unsafe_allow_html=True)
    map_col, bar_col = st.columns([2,1], gap='small')
    with map_col.expander(f'Cityscape of {brand_name}: Unveiling Customer Hotspots', expanded=True):
        st.pydeck_chart(customer_cities_map, use_container_width=True)
    with bar_col.expander(f"City Spotlight: {brand_name}'s Top Customer Cities", expanded=True):
        st.altair_chart(customer_cities_bar, use_container_width=True)
    
with tam_tab:
    selection_col, rank_col = st.columns(2, gap='small')
    first_store_stat_col, second_store_stat_col = st.columns(2, gap='small')
    st.divider()
   
    # create two columns for the map and bar chart
    map_col, checkbox_col = st.columns([2.5,1], gap='small')
    
    # defaults
    mapstyle = "mapbox://styles/mapbox/dark-v11"
    layers_list = []
    initial_lng = prime_store_lng
    initial_lat = prime_store_lat
    zoom=10
    
    # create checkboxes for the different available views
    checkbox_col.markdown('<font style="font-size:12px;">Select the features you want to see on the map</font>',
                     unsafe_allow_html=True)
    neighborhoods_view = checkbox_col.checkbox('Neighborhoods', True)
    stores_view = checkbox_col.checkbox('Available stores', True)
    dt_view = checkbox_col.checkbox('Distance and time from neighborhoods to store')
    highways_view = checkbox_col.checkbox('Highways near store')
    busstop_view = checkbox_col.checkbox('Bus stops near store')
    business_comm_view = checkbox_col.checkbox('Notable commercial businesses near store')
    parking_view = checkbox_col.checkbox('Public parking spaces near store')
    
    # select the store to view and change the store stats and map accordingly
    selected_store = selection_col.selectbox('Select a store to inspect', store_locations)
    
    selected_choice = store_stats.loc[store_stats.index == selected_store, 'choice'].iloc[0]
    if selected_choice == 'best': emoji = '🟢'; text = 'Best Choice'
    elif selected_choice == 'good': emoji = '🟡'; text = 'Good Choice'
    elif selected_choice == 'fair': emoji = '🟠'; text = 'Fair Choice'
    else: emoji = '🔴'; text = 'Bad Choice'

    selected_store_stats = store_stats.loc[store_stats.index == selected_store]
    selected_store_pop = pop_df.loc[:, [selected_store]]
    selected_store_pop = selected_store_pop.loc[selected_store_pop[selected_store] == selected_store_pop[selected_store].max()].index[0]
    
    selected_store_data = prepped_store_data.loc[prepped_store_data['store_location'] == selected_store, :].reset_index()
    initial_lng, initial_lat = selected_store_data.loc[0, 'lng'], selected_store_data.loc[0, 'lat']
    
    selected_other_stores_data = other_stores_data[other_stores_data['store_location'] == selected_store]
    selected_dt_data = path_df[path_df['store'] == selected_store]

    # create the layer for the neighborhoods
    neighborhoods_layer = pdk.Layer(type='PolygonLayer', data=prime_community,
                                    get_polygon='geometry', pickable=True, filled=False,
                                    get_line_color=list(udfs.hex_to_rgb(primary_color)) + [160],
                                    get_line_width=75, auto_highlight=True
                                   )
    
    # create the layer for the polygon of the stores
    stores_polygon_layer = pdk.Layer(type='PolygonLayer', data=store_data, get_polygon='geopolygon', filled=True,
                                     get_fill_color=list(udfs.hex_to_rgb(features_color)) + [160],
                                     get_line_color=list(udfs.hex_to_rgb(features_color)),
                                     get_line_width=10, pickable=True, auto_highlight=True, extruded=False
                                     )
    
    # create the layer for the prime store icon
    prime_store_icon_layer = pdk.Layer(type="IconLayer", data=prime_store_data, get_icon="icon_data", get_size=2,
                                       size_scale=15, get_position=["lng", "lat"], pickable=True,
                                       auto_highlight=True
                                       )
    
    # create the layer for the other stores icons
    other_store_icon_layer = pdk.Layer(type="IconLayer", data=other_stores_data, get_icon="icon_data", get_size=1.25,
                                       size_scale=15, get_position=["lng", "lat"], pickable=True,
                                       auto_highlight=True
                                       )
    
    # create layer for distance and time
    dt_layer = pdk.Layer(type='PathLayer', data=selected_dt_data, pickable=True, get_color='color', width_scale=10,
                         width_min_pixels=2, get_path='path', get_width=1, auto_highlight=True
                         )
    
    # create the layer for commercial businesses
    business_comm_layer = pdk.Layer(type="IconLayer", data=business_comm, get_icon="icon_data", get_size=1.25,
                                    size_scale=15, get_position=["lng", "lat"], pickable=True,
                                    auto_highlight=True
                                    )
    
    # create the layer for highways
    highways_layer = pdk.Layer(type='PathLayer', data=highways, pickable=True, 
                               get_color=list(udfs.hex_to_rgb(features_color)), width_scale=10,
                               width_min_pixels=2, get_path='geometry', get_width=1, auto_highlight=True
                               )
    
    # create the layer for busstops
    busstop_layer = pdk.Layer(type="IconLayer", data=busstop, get_icon="icon_data", get_size=0.75,
                                    size_scale=15, get_position=["lng", "lat"], pickable=True,
                                    auto_highlight=True)
    
    # create the layer for public parking spaces
    parking_layer = pdk.Layer(type="IconLayer", data=parking, get_icon="icon_data", get_size=1,
                              size_scale=15, get_position=["lng", "lat"], pickable=True, auto_highlight=True)
    
    # activate the checkboxes
    if neighborhoods_view:
        layers_list.append(neighborhoods_layer)
        tooltip={"text": """Neighborhood: {neighborhoods}\nExisting customers: {population_string}"""}
        zoom = 10
    
    if stores_view:
        layers_list.append(stores_polygon_layer)
        layers_list.append(prime_store_icon_layer); layers_list.append(other_store_icon_layer)
        tooltip={"text": """Store: {name}\nAddress: {address}"""}
        zoom=13
        
    if dt_view:
        layers_list.remove(other_store_icon_layer); layers_list.append(dt_layer)
        other_store_icon_layer = pdk.Layer(type="IconLayer", data=selected_other_stores_data, get_icon="icon_data",
                                           get_size=1.25, size_scale=15, get_position=["lng", "lat"], pickable=True,
                                           auto_highlight=True
                                          )
        tooltip={'text': """From: {neighborhoods}\nTo: {store}\nTime: {time} mins\nDistance: {distance} km"""}
        layers_list.append(other_store_icon_layer); zoom=10

    if highways_view:
        layers_list.append(highways_layer)
        tooltip={'text': """Name: {name}"""}
        zoom=12

    if busstop_view:
        layers_list.append(busstop_layer)
        tooltip={'text': """Name: {name}\nAlternative name: {alt_name}"""}
        zoom=12

    if business_comm_view:
        layers_list.append(business_comm_layer)
        tooltip={'text': """Name: {name}"""}
        zoom=12

    if parking_view:
        layers_list.append(parking_layer)
        tooltip={'text': """Name: {name}"""}
        zoom=12
    
    with rank_col.expander('Rank', expanded=True):
        st.markdown(f'''{emoji} {text}''')

    first_store_stat_col.markdown(f"""
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Store name: </div>
            <div style='text-align: right;'>{selected_store}</div> 
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Store size: </div>
            <div style='text-align: right;'>{selected_store_stats['store_size'].iloc[0]:,} sqm</div> 
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Avg distance from customer neighborhoods: </div>
            <div style='text-align: right;'>{selected_store_stats['avg_distance'].iloc[0]} km</div> 
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Avg time spent in traffic: </div>
            <div style='text-align: right;'>{selected_store_stats['avg_travel_time'].iloc[0]} mins</div> 
        </div>
        </div><div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Number of nearby major roads: </div>
            <div style='text-align: right;'>{selected_store_stats['highways'].iloc[0]}</div> 
        </div>
        """,
        unsafe_allow_html=True)

    second_store_stat_col.markdown(f"""
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Design index: </div>
            <div style='text-align: right;'>{selected_store_stats['design'].iloc[0]}%</div> 
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Accessibility index: </div>
            <div style='text-align: right;'>{selected_store_stats['accessibility'].iloc[0]}%</div> 
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Available parking spaces: </div>
            <div style='text-align: right;'>{selected_store_stats['parking_space'].iloc[0]}</div> 
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Predicted walk-ins: </div>
            <div style='text-align: right;'>{population[selected_store]:,} customers</div> 
        </div>
        </div><div style='display: flex; justify-content: space-between; font-size: 12px;'>
            <div style='text-align: left; font-weight: bold;'>Most frequent neighborhood: </div>
            <div style='text-align: right;'>{selected_store_pop}</div> 
        </div>
        """,
        unsafe_allow_html=True)

    # set the initial view state
    initial_view_state = pdk.ViewState(longitude=initial_lng, latitude=initial_lat, 
                                       zoom=zoom, min_zoom=2, pitch=50)
    
    with map_col.expander(f'**Locate, Shop, Commute:** Snapshots of Customer Neighborhood Hotspots, Stores Proximity, and Surroundings',
                          expanded=True):
        trade_area_map = pdk.Deck(map_style=mapstyle, layers=layers_list,
                                  initial_view_state=initial_view_state, tooltip=tooltip
                                  )
        st.pydeck_chart(trade_area_map, use_container_width=True)
