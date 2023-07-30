import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
import altair as alt

st.set_page_config(page_title='Customer Segmentation', page_icon=Image.open('media/images/people.png'), layout='wide')

st.markdown('<h1 style="color:#8e43e7; font-size:28px; font-weight:bold">💡 Customer Persona Insights</h1>',
            unsafe_allow_html=True)

st.markdown(f'''
            <p style="font-size: 20px; color:#afafaf; line-height:2;">
            The basic insights for each persona group is shown below. You can request to download these insights as either a <strong>PowerBI, Tableau or Google Looker Studio file</strong> for free. You can also request for additional insights such as the EDA for each persona.<br>
            </p>
            <span style='background-color: #8e43e7; border-radius: 8px; padding-left: 8px; padding-top: 8px; padding-bottom: 10px; padding-right: 8px; color: #cfcfcf; font-size: 20px;'><a href="mailto:itunu.owo@gmail.com" style="color:#cfcfcf"> Request additional insights</a></span>&emsp;
            <span style='border: 2px solid #8e43e7; border-radius: 8px; padding-left: 8px; padding-top: 8px; padding-bottom: 10px; padding-right: 8px; color: #8e43e7; font-size: 20px;'><a href="mailto:itunu.owo@gmail.com" style="color:#8e43e7"> Download basic insights</a></span>
            </p><br>            
            ''', unsafe_allow_html=True)

try:
    rfm_categories_df = st.session_state['rfm_categories_df']
    rfm_scores_df = st.session_state['rfm_scores_df']
    rfm_df = st.session_state['rfm_df']
    cust_column = st.session_state['cust_column']

    col1, col2 = st.columns(2)
    cust_num_chart_df = rfm_df.groupby(['RFM Category', 'Persona']).count()[cust_column].sort_values(ascending=False).reset_index().rename(columns={cust_column:'Number of Customers'})

    def create_chart(df=cust_num_chart_df, xaxis='Number of Customers', yaxis='Persona', add_tip='RFM Category', sort_by = list(cust_num_chart_df['Persona'])):
        chart = (alt.Chart(df, height=280)
                .mark_bar(color='#8e43e7')
                .encode(x=alt.X(xaxis, axis=alt.Axis(labels=False)),
                        y=alt.Y(yaxis, axis=alt.Axis(title=None), sort=sort_by),
                        tooltip=[xaxis, yaxis, add_tip])
                )
        text = chart.mark_text(align='left', baseline='middle', dx=3, color='#cfcfcf').encode(text=xaxis)
        return chart+text

    cust_num_chart = create_chart()
    with col1.expander('Number of Customers in the Different Persona Groups', expanded=True):
        st.altair_chart(cust_num_chart,use_container_width=True)

    median_df = round(rfm_df.groupby(['RFM Category', 'Persona']).median(numeric_only=True).reset_index(),2)

    recency_chart = create_chart(df=median_df, xaxis='Recency')
    with col2.expander('Median Number of Days Since Last Order in the Different Persona Groups', expanded=True):
        st.altair_chart(recency_chart,use_container_width=True)

    frequency_chart = create_chart(median_df, 'Frequency')
    with col1.expander('Median Number of Orders in the Different Persona Groups', expanded=True):
        st.altair_chart(frequency_chart,use_container_width=True)

    monetary_chart = create_chart(median_df, 'MonetaryValue')
    with col2.expander('Median Order Value ($) in the Different Persona Groups', expanded=True):
        st.altair_chart(monetary_chart,use_container_width=True)
    
    for key in st.session_state.keys():
        del st.session_state[key]
except:
    st.warning('There were no insights to generate. Have you created personas in the previous page yet?')