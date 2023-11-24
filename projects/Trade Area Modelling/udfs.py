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
