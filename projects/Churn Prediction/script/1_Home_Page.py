import functions as udfs
import streamlit as st

try:
    logo = udfs.logo

    # set page configuration
    st.set_page_config(page_title='Churn Prediction', page_icon=logo, layout='wide',
                       initial_sidebar_state='collapsed')

    # show toast message
    st.toast('This app is best viewed on a PC 😊')

    # header
    st.title('Churn Prediction')

    udfs.display_intro_columns()

    if st.session_state["Investigate Further"]:
        st.switch_page('pages/Analysis_Insights.py')
    elif st.session_state["Training and Testing"]:
        st.switch_page('pages/Churn_Model.py')
    elif st.session_state["View Actions"]:
        st.switch_page('pages/Recommended_Actions.py')

except Exception as error:
    st.warning(f'An error occured on line {error.__traceback__.tb_lineno}: {error}')