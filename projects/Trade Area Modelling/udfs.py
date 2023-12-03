from shapely import wkt
import streamlit as st
import altair as alt
import pandas as pd

# read csv file
@st.cache_data(ttl=12*60*60)
def load_data(file):
    df = pd.read_csv(file)
    return df

# conditionally color bar charts
def bar_conditional_color(barchart, words_list, field, color):
    bar_color = barchart.transform_filter(
            alt.FieldOneOfPredicate(field=field, oneOf=words_list)
        ).encode(color=alt.value(color))
    return bar_color

# create bar charts
def altair_bar_chart(df, xaxis, xtitle, yaxis, textaxis, tooltip, color):
    # create bar chart and encode text
    bar_chart = alt.Chart(df).mark_bar(color=color)\
                .encode(
                        x=alt.X(xaxis, axis=alt.Axis(labels=False, title=xtitle)),
                        y=alt.Y(yaxis, axis=alt.Axis(title=None, labelLimit=180), sort=df[xaxis].tolist()),
                        tooltip=tooltip
                        )
    text = bar_chart.mark_text(align='left', color=color, baseline='middle', dx=3).encode(text=textaxis)
    bar_chart = bar_chart + text
    return bar_chart

# convert a Polygon string to a list of coordinates
def wkt_to_coordinates(wkt_geom):
    geom = wkt.loads(wkt_geom)
    if geom.is_empty: return None
    if geom.geom_type == 'Polygon':
        coordinates = [list(geom.exterior.coords)]
    elif geom.geom_type == 'LineString':
        coordinates = list(geom.coords)
    elif geom.geom_type == 'MultiPolygon':
        coordinates = [list(p.exterior.coords) for p in geom.geoms]
    else:
        return None
    return coordinates

# convert hex color to rgb
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

# build trade area model
def trade_area_model(customer_communities, store_data, store_distances, store_preferences):
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
    pop_df = pop_df.astype(int)
    population = pop_df.sum(axis=0).to_dict()

    # select the store with the max probable sales
    max_population = max(population.values())
    optimum_store = [key for key in population.keys() if population[key] == max_population][0]
    return pop_df, population, max_population, optimum_store

# add icon dictionary to df
def add_icon_to_df(df, icon_properties, icon_url):
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df['lng'] = df['geometry'].apply(lambda x: x.centroid.x)
    df['lat'] = df['geometry'].apply(lambda x: x.centroid.y)
    df.drop('geometry',axis=1,inplace=True)
    icon_data = icon_properties.copy()
    icon_data['url'] = icon_url
    df['icon_data'] = [icon_data] * len(df)
    return df

