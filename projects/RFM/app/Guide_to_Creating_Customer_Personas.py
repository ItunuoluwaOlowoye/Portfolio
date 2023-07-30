import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(page_title='Customer Segmentation', page_icon=Image.open('media/images/people.png'), layout='wide')

rfm_categories = pd.read_excel('datasets/rfm.xlsx', sheet_name='categories')
rfm_scores = pd.read_excel('datasets/rfm.xlsx', sheet_name='scores')

st.markdown('<p style="color:#8e43e7; font-size:28px; font-weight:bold">📖 Understanding your Customers: Guide to Creating Customer Personas using RFM</p>',
            unsafe_allow_html=True)

st.write('')
col1, col2 = st.columns(2)
col1.markdown(f'''<h3> What is RFM Analysis? </h3>
            <p style="font-size: 20px; color:#afafaf; line-height:2; display: flex; justify-content: space-between">
            When running a business, you would have different types of customers; premium, loyal, one-off, active customers etc.<br>
            RFM (Recency, Frequency, and Monetary value) is a technique used to group customers based on their purchasing behaviour. The different groups can then be targeted with marketing campaigns tailored to their needs.<br><br>
            </p>
            ''', unsafe_allow_html=True)
col2.video('media/videos/who is your customer.mp4')
st.markdown(f'''<h3 style='text-align: center;'> Using this app for RFM analysis </h3>
            <p style="font-size: 20px; line-height:2">
            You can use this app to analyze your sales/orders/transactions from customers on the <a href='http://localhost:8501/Creating_Customer_Personas' style='color:#8e43e7'>'Creating Customer Personas' page</a>. We do not keep records of your data in our database. <strong> As an added layer of security, you can also tokenize any personal information present in the data. </strong>A sample dataset has been provided for guidance.
            </p>
            ''', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.write('')
col1.video('media/videos/data cleaning.mp4')
col2.markdown(f'''
            <h4> Cleaning the Data </h4>
            <p style="font-size: 20px; color:#afafaf; line-height:2">
            Your dataset must be cleaned and pre-processed. Note:<br>
            1. There is a column for customer identity (e.g customer ID), order number, order date, and revenue from the order in the table.<br>
            2. The date column is formatted as a date with consistent format<br>
            3. The revenue column is formatted as a number<br>
            <span style='background-color: #8e43e7; border-radius: 8px; padding-left: 8px; padding-top: 8px; padding-bottom: 10px; padding-right: 8px; color: #cfcfcf; font-size: 20px;'>❗<a href="mailto:itunu.owo@gmail.com" style="color:#cfcfcf"> Need help tokenizing and cleaning your data?</a></span>
            </p>
            ''', unsafe_allow_html=True)
st.markdown(f'''
            <h4 style='text-align:center'> Exploring the Data (EDA) </h4>
            <p style="font-size: 20px; color:#afafaf; line-height:2">
            Understanding your customers' purchasing behaviours also require reports about the business operations such as number of customers, number of orders, revenue generated per day, orders created per day, orders completed per day, products/services sold, revenue per product etc.<br>
            The EDA of the sample data is shown on the <a href='http://localhost:8501/Exploring_Your_Data' style='color:#8e43e7'>'Exploring Your Data' page</a>.
            </p>
            ''',unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.markdown(f'''
            <h4> Calculating RFM and Creating Personas </h4>
            <p style="font-size: 20px; color:#afafaf; line-height:2">
            RFM is calculated this way:<br>
            1. Recency: How long ago an order was placed by the customers in the given period of time with the most recent date as the benchmark. High recency means not too long ago and vice versa<br>
            2. Frequency: How many orders were made by the customers in the given period of time. High frequency means high number of orders and vice versa<br>
            3. Monetary Value: How much, on average, was spent by the customer per order in the given period of time. High monetary value means high price and vice versa<br>
            ''',unsafe_allow_html=True)
col2.title(''); col2.title('')
col2.video('media/videos/persona.mp4')
st.markdown(f'''<p style="font-size: 20px; color:#afafaf; line-height:2">
            <strong> Real data is rarely symmetric. The median was used as an alternative to the average since it is a better statistic for skewed data.</strong>
            The recency, frequency, and monetary value are ranked on a scale of 0 to 1 each. To create categories, 0 to 0.5 is ranked as Low while above 0.5 is ranked as High. To create scores, 0 to 0.5 is assigned a value of 0 while above 0.5 is assigned a value of 1. Personas and segments are created from a combination of these categories and scores as shown below:
            </p>
            ''', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.write(rfm_categories.set_index('Persona'))
col2.write(rfm_scores.set_index('Segment'))
st.markdown(f'''<h4 style='text-align:center'> Persona Insights </h4>
            <p style="font-size: 20px; color:#afafaf; line-height:2">
            For each persona group, a report/dashboard is created showing:<br>
            1. The number of customers<br>
            2. The recency, frequency, and monetary value statistics<br>
            The persona insights of the sample data is shown on the <a href='http://localhost:8501/Customer_Persona_Insights' style='color:#8e43e7'>'Customer Persona Insights' page</a>.
            </p>
            ''', unsafe_allow_html=True)
