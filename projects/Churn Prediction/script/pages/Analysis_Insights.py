import functions as udfs
import streamlit as st

if 'app'=='app':
    st.page_link(page='1_Home_Page.py', icon='⬅', label='Back to Home Page')

    telco_df = udfs.telco_df

    udfs.analysis_insights(df=telco_df)

#except Exception as error:
 #   st.warning(f'An error occured on line {error.__traceback__.tb_lineno}: {error}')