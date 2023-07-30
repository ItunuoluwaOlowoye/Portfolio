import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
import altair as alt

st.set_page_config(page_title='Customer Segmentation', page_icon=Image.open('media/images/people.png'), layout='wide')

st.markdown('<h1 style="color:#8e43e7; font-size:28px; font-weight:bold">🏷️Creating Customer Personas</h1>',
            unsafe_allow_html=True)

@st.cache_data(ttl=1*60*60)
def read_rfm_data(file='datasets/rfm.xlsx', category_sheet = 'categories', category_cols = ['RFM Category', 'Persona'], score_sheet = 'scores', score_cols = ['RFM Score', 'Segment']):
    categories_df = pd.read_excel(file, sheet_name=category_sheet, usecols=category_cols)
    scores_df = pd.read_excel(file, sheet_name=score_sheet, usecols=score_cols).drop_duplicates()
    return categories_df, scores_df
rfm_categories_df, rfm_scores_df = read_rfm_data()

col1, col2 = st.columns(2, gap='large')

with col1:
    option = st.radio('Choose preferred option', ['Upload your own file', 'Run with sample data'], horizontal=True)
    if option == 'Run with sample data':
        df = pd.read_csv('datasets/superstore_sales.csv', parse_dates=['Order Date', 'Ship Date'])
        cust_column = 'Customer ID'
        date_column = 'Order Date'
        freq_column = 'Order ID'
        monet_column = 'Sales'
        st.markdown('Data preview...', unsafe_allow_html=True)
        st.write(df.set_index('Row ID').head())
        st.download_button('Download full dataset as CSV here', data=df.to_csv(index=False), file_name='superstore_sales.csv', mime='text/csv')
    else:
        file = st.file_uploader('Upload file')
        if file is not None:
            try:
                file_type = st.radio('What is the file type?', options=['csv','xlsx'], horizontal=True)
                col3, col4 = st.columns(2, gap='medium')
                if file_type == 'csv': df = pd.read_csv(file)
                else: df = pd.read_excel(file)
                df_columns = [''] + list(df.columns)
                cust_column = col3.selectbox('Select the column for customers', options=df_columns)
                freq_column = col3.selectbox('Select the column for orders', options=df_columns)
                monet_column = col4.selectbox('Select the column for order price', options=df_columns)
                date_column = col4.selectbox('Select the column for order dates', options=df_columns)
                df[date_column] = pd.to_datetime(df[date_column], errors='raise')
            except: st.error('Select valid columns')
    try:
        max_date = df[date_column].max()
        rf_df = df.groupby(cust_column).agg(Recency = (date_column, 'max'), Frequency = (freq_column, 'nunique'))
        rf_df['Recency'] = (max_date - rf_df['Recency']).dt.days
        m_df = df.groupby([cust_column, freq_column]).sum(numeric_only=True).reset_index().groupby(cust_column).agg(MonetaryValue = (monet_column, 'median'))
        rfm_df = pd.concat([rf_df, m_df], axis=1)
        r_rank = rfm_df['Recency'].rank(pct=True, ascending=False)
        f_rank = rfm_df['Frequency'].rank(pct=True)
        m_rank = rfm_df['MonetaryValue'].rank(pct=True)
        ranks = [r_rank, f_rank, m_rank]
        bins=[0,0.5,1]; names= ['L','H']; numbers = [0,1]
        categories, scores = ({} for _ in range(2))
        for rank in ranks:
            category = pd.cut(rank, bins, labels=names)
            score = pd.cut(rank, bins, labels=numbers)
            categories[rank.name] = category.astype(str)
            scores[rank.name] = score.astype(int)
        rfm_df['RFM Category'] = categories['Recency'] + categories['Frequency'] + categories['MonetaryValue']
        rfm_df['RFM Score'] = scores['Recency'] + scores['Frequency'] + scores['MonetaryValue']
        rfm_df.reset_index(inplace=True)
        rfm_df = rfm_df.merge(rfm_categories_df, how='left', on='RFM Category').merge(rfm_scores_df, how='left', on='RFM Score')
        download_df = rfm_df.rename(columns={'Recency':'DaysSinceLastDate', 'Frequency':'NumberOfOrders', 'MonetaryValue':'AverageOrderSale'})
        col5, col6 = st.columns(2, gap='medium')
        col5.download_button('Download personas as CSV', data=download_df.to_csv(index=False), file_name='personas.csv')
        buffer = BytesIO()
        writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
        download_df.to_excel(writer, engine='xlsxwriter', index=False)
        writer.close()
        col6.download_button(label="Download personas as XLSX", data=buffer, file_name=f'personas.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') # download file
    except: st.warning('Have you uploaded your file, and selected the correct file extension and columns?')

with col2:
    try:
        name_col, filter_col = st.columns([1,3])
        customer_id_list = rfm_df[cust_column].unique()
        name_col.write('Select a customer')
        customer = filter_col.selectbox(cust_column, options=customer_id_list, label_visibility='collapsed')
        df_filter = rfm_df[rfm_df[cust_column]==customer].fillna('')
        recency = df_filter.iloc[0]['Recency']
        frequency = df_filter.iloc[0]['Frequency']
        monet = round(df_filter.iloc[0]['MonetaryValue'],2)
        persona = df_filter.iloc[0]['Persona']
        segment = df_filter.iloc[0]['Segment']
        if segment == 'Bronze': segment = '🥉'
        elif segment == 'Silver': segment = '🥈'
        elif segment == 'Gold': segment = '🥇'
        else: segment = ''

        st.markdown(
            f"""
            <h3 style="color:#8e43e7; padding-left: 8px; font-weight:bold;">{segment} {customer}</h3>
            <h5 style="padding-left: 8px; ">Recency, Frequency, and Monetary Value (RFM) Analysis</h5>
            <div style='display: flex; justify-content: space-between; font-size: 20px;'>
                <div style='text-align: left; padding-left: 8px;'>Recency: </div>
                <div style='text-align: right; padding-right: 8px;'><strong>{recency} days</strong> since last recorded date</div>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: 20px;'>
                <div style='text-align: left; padding-left: 8px;'>Frequency: </div>
                <div style='text-align: right; padding-right: 8px;'>{frequency} orders</div>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: 20px; padding-bottom:20px'>
                <div style='text-align: left; padding-left: 8px;'>Monetary Value: </div>
                <div style='text-align: right; padding-right: 8px;'><strong>${monet}</strong> median order value</div>
            </div>
            <div style='background-color: #8e43e7; border-radius: 8px; padding-left: 8px; padding-top: 8px; padding-bottom: 8px; color: #cfcfcf; font-size: 28px; font-weight:bold'>{persona} Customer</div>
            """,
            unsafe_allow_html=True)        
    except: pass

try:
    st.session_state['rfm_categories_df'] = rfm_categories_df
    st.session_state['rfm_scores_df'] = rfm_scores_df
    st.session_state['rfm_df'] = rfm_df
    st.session_state['cust_column'] = cust_column
except: pass