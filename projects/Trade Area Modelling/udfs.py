from shapely import wkt
import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
import numpy as np
import os

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
def wkt_polygon_to_coordinates(wkt_polygon):
    polygon = wkt.loads(wkt_polygon)
    if polygon.is_empty: return None
    if polygon.geom_type == 'Polygon':
        coordinates = [list(polygon.exterior.coords)]
    elif polygon.geom_type == 'LineString':
        coordinates = list(polygon.coords)
    elif polygon.geom_type == 'MultiPolygon':
        coordinates = [list(p.exterior.coords) for p in polygon.geoms]
    else:
        return None
    return coordinates

# convert hex color to rgb
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
