from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from sklearn.preprocessing import StandardScaler
from django.contrib.auth import authenticate
from langchain.agents import AgentType
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from wordcloud import WordCloud
from typing import Union
import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
import numpy as np
import squarify
import os

############### DEFAULT VARIABLES ####################

logo = Image.open('pictures/logo.png')
random_seed = 42
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}
store_name = 'Emporium'
primary_color = '#2F0A0D'
text_color = '#524748'
secondary_color = '#3E0E12'
tertiary_color = '#7B6F71'
grey_color = '#E7E5E5'
purchase_beh_segment = {
   'Value Seekers': ['prioritize stretching their budget', """For them, think coupons, clearance sales,
                     budget-friendly brands, and subscription programs"""],
   'Luxury Seekers': ['crave high-quality treats', """We'll roll out the red carpet with exclusive product launches,
                      high-end products, and limited time offers on premium brands"""],
   'Savvy Spenders': ['value both quality and strategic spending', """They'll enjoy loyalty program perks and 
                      curated collections of package deals"""],
   'Bargain Hunters': ['actively search for bargains and the best deals', """Think clearance sales, limited-time 
                       discounts, and bulk purchase discounts"""]
}
loyalty_segment = {
    'Wanderers': ['either shop with many competitors or spend more at competitors', """We'll lure
                  them back with price-matching guarantees, competitive deals, and strategic product placements"""],
    'Explorers': ['are open to venturing out', """Special occasion promotions like holiday sales and 
                  limited-edition products will woo them"""],
    'Loyalists': ['are consistent customers', """They'll be rewarded with loyalty program perks, early access to
                  new products and promotions, and exclusive community events"""]
}
product_pref_segment = {
    'Universal Shoppers': ['frequently shop across product categories', """We'll dish up cross-category discounts
                           and early access to exciting new arrivals"""],
    'Category Hoppers': ['are open to shopping different product categories', """They'll love bundled deals across
                         categories and rewards for diversified purchases"""],
    'Style & Sustenance Seekers': ['have groceries and fashion interests', """Promotions and tutorial subscriptions
                                   on groceries and fashion will interest them"""],
    'Pantry Planners': ['prioritize stocking up on groceries', """We'll offer them curated promotions on groceries
                        and expert advice subscriptions"""]
}
channel_pref_segment = {
    'Digital Crew': ['love digital convenience and efficiency', """We'll offer exclusive online deals on both the
                     website and the mobile app"""],
    'Omnichannel Crew': ["""seamlessly hop between shopping channels for all categories""", """We'll give them
                         cross-channel offers and rewards for multichannel purchases"""],
    'Brick & Mortar Crew': ['cherish the in-store experience', """They'll feel pampered with engaging in-store
                            events and promotions"""],
    'Click & Brick Crew': ["""have split shopping preferences: groceries in-store and others online""", """They'll
                           enjoy cross-channel promotions and in-store pickups for online purchases"""]}
access_pages = {
    'developer': '1_🛡️_Master_Page.py',
    'marketing': '2_🛍️_Customer_Profiles.py',
    'serviceagent': '11_🤖_Customer_Service_Agent.py'
}
num_freq_groups = {0:'Never', 2:'Once or twice', 3:'Monthly', 6:'Every two weeks', 
                   12:'Weekly', 36:'More than once per week'}
app_conversion = {'Converted':'uses the app to shop', 'Churned':'has stopped using the app to shop',
                  'Not used':'has not used the app before'}


############### LAYOUT FUNCTIONS ####################

# django authentication step 1
def password_entered():
    # authenticate user with Django
    user = authenticate(username=st.session_state['username'], password=st.session_state['password'])
    # check if pasword is correct in session state
    if user is not None:
        st.session_state.clear()
        st.session_state.user = user
    else: 
        st.warning('⚠️ Oops! You have entered either a wrong username or wrong password.')
    
# django authentication step 2
def authenticate_user():
    # if password is not correct, create text inputs for username and password
    st.text_input('Username (case-sensitive)', key='username')
    st.text_input('Password', type='password', key='password')
    # check username and password when button is pressed
    if st.button('Log in', type='primary'):
        password_entered()
        # switch page for the respective access levels if authenticated
        if 'user' in st.session_state:
            page = access_pages.get(st.session_state.user.username, '')
            st.switch_page(page=f'pages/{page}')
        else: st.stop()
    # else stop execution
    else: st.stop()

# resize image
def resize_image(_image:Image, new_image_height:float):
    # set the original image size to a variable
    image_size = _image.size
    # separate the width and height
    image_width, image_height = (image_size[0], image_size[1])
    # calculate the ratio between the original and new image height
    ratio = image_height/new_image_height
    # calculate the new image width with the ratio
    new_image_width = int(image_width/ratio)
    # return the newly resized image
    new_image = _image.resize((new_image_width,new_image_height))
    
    return new_image

# default home page intro
def home_page():
    # create content
    marketing = f"""<div style='font-size:12px;'>
    <font style='font-size:14px;'>Marketing teams</font> need customer segments to
    <strong style='color:{primary_color};'>tailor marketing campaigns</strong> towards customers' needs,
    <strong style='color:{primary_color};'>offer personalized recommendations</strong> to meet those needs,
    develop strategies to <strong style='color:{primary_color};'>improve customer loyalty</strong>,and
    <strong style='color:{primary_color};'>improve omnichannel experiences</strong> (i.e., in-store, website,
    and mobile app experiences) for customers.<br>
    </div>"""
    service_agent = f"""<div style='font-size:12px;'>
            <font style='font-size:14px;'>Customer service agents</font> need customer personas to improve
            customer service experience. They can attend to the customers' immediate needs, use personas to
            know which segments customers belong to, and <strong style='color:{primary_color};'>offer 
            personalized suggestions</strong> to satisfy other needs and 
            <strong style='color:{primary_color};'>increase upselling opportunities.</strong><br>
            </div>"""

    # create columns for content
    cols = st.columns(2, gap='medium')

    # for each login section
    for col, section, button_text in zip(cols, (marketing, service_agent),
                                         ('marketing', 'service_agent')):
        with new_container(col):
            st.markdown(section, unsafe_allow_html=True)
            st.image(f'pictures/{button_text}.jpg')
            text = f"Go to {button_text.replace('_',' ')} page"
            st.session_state[button_text] = st.button(text, type="primary", use_container_width=True)

# customer profiles intro columns
def display_customer_profiles_intro_columns():
    # create content
    executive_summary = f"""{store_name} is a department store that wants to improve customer
    experience and develop strategic marketing campaigns that resonate with their customer base."""
    segmentation_insights = """We analysed survey data containing customers' biodata, purchasing 
    behaviours, interactions with competitors, preferred products, and preferred channel experiences."""
    customer_personas = """We created personas based on the customer segments and used these personas to 
    personalise recommended marketing campaigns for each customer"""

    # Create columns for content
    cols = st.columns(3)
    # for each section, create contact card with icon, title, body, and action button
    for col, icon, title, content, button_text in zip(
        cols,
        ("📑", "💡", "🧑‍🦱"),
        ("Executive Summary", "Segmentation Insights", "Customer Personas"),
        (executive_summary, segmentation_insights, customer_personas),
        ("Further Reading", "Investigate Further", "View Personas"),
    ):
        with col.container(border=True, height=450):
            st.markdown(f"""
                        <p style="font-size:14px;">
                        <strong>{icon} {title}</strong><br><br>
                        {content}
                        </p>""",
                        unsafe_allow_html=True,
            )
            image = resize_image(_image=Image.open(f'pictures/{title.lower().replace(" ", "_")}.png'),
                                 new_image_height=200)
            st.image(image)
            st.session_state[button_text] = st.button(button_text, type="primary", use_container_width=True)

# create word cloud
@st.cache_data(ttl=1*60*60)
def plt_word_cloud(df:pd.DataFrame, xaxis:str, yaxis:str, colored_words_list:Union[list, None],
                   default_color:str):
    # define a color function
    def color_func_(word, font_size, position, orientation, random_state=None, **kwargs):
        return primary_color if word in colored_words_list else default_color
    # generate wordcloud
    wordcloud_fig = plt.figure(figsize=(20, 10))
    # create word cloud in a pyplot figure
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_font_size=40,
                          color_func=color_func_, random_state=random_seed)\
                            .generate_from_frequencies(df.set_index(xaxis)[yaxis])
    plt.imshow(wordcloud)
    plt.axis("off")
    return wordcloud_fig

# executive summary
@st.cache_data(ttl=1*60*60)
def display_executive_summary(df:pd.DataFrame):
    # intro
    st.markdown(f"""
                <p style='font-size:14px;'>
                This executive summary paints a picture of {store_name}'s unique customer *segments, revealing
                their quirks and preferences, which can be used to craft personalized shopping experiences.<br>
                    <font style='font-size:12px;'>
                        *Based on <strong style='color:{primary_color};'>purchasing habits, loyalty and
                        competitor landscape, preferred product categories, and preferred shopping
                        methods</strong>.
                    </font>
                </p>
                """, unsafe_allow_html=True)

    st.divider()
    # build word cloud data  
    wordcloud_df = pd.concat([df['behavioural_persona'], df['competitor_landscape_persona'], df['product_persona'],
                              df['channel_persona']], axis=0).value_counts().to_frame().reset_index()
    
    # create lists and zip them relating each title to its segment
    titles = ['Purchasing Habits', 'Loyalty and Competitor Landscape', 'Product Categories',
              'Shopping Channels']
    segment_list = [purchase_beh_segment, loyalty_segment, product_pref_segment, channel_pref_segment]
    for title, segment_dict_ in zip(titles, segment_list):
            st.markdown(f"""
                        <strong style='font-size:18px; padding-bottom:5px; color:{secondary_color}'>{title}</strong>
                        """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            colored_words = [key.removesuffix('s') for key in segment_dict_.keys()]
            # build word cloud with color emphasizing each segment    
            wordcloud_fig = plt_word_cloud(df=wordcloud_df, xaxis='index', yaxis='count',
                                           colored_words_list=colored_words, default_color=grey_color)
            # brief findings section
            segment_variables = zip(segment_dict_.keys(), [val[0] for val in segment_dict_.values()],
                                    [val[1] for val in segment_dict_.values()])
            if segment_dict_ == purchase_beh_segment or segment_dict_ == product_pref_segment:
                column1 = col1; column2 = col2
            else: column1 = col2; column2 = col1
            column1.pyplot(wordcloud_fig)
            for name, profile, insight in segment_variables:
                column2.markdown(f""" <div style='font-size:14px;'>
                              <strong style='color:{primary_color}'>{name}</strong> {profile}. {insight}.
                              </div>""", unsafe_allow_html=True)
            st.divider()

# create a new container
def new_container(command:st, border:bool=True, height:Union[int, None]=None):
    return command.container(border=border, height=height)

# univariate bar chart
@st.cache_data(ttl=1*60*60)
def univariate_bar_chart(df:pd.DataFrame, column_name:str, title:str, scale_domain:list = [0, 0.5],
                         chart_height:int=200, color:str=grey_color):
    # count attribute occurence
    transformed_df = df[column_name].value_counts(normalize=True)
    # round to two decimal places
    transformed_df = round(transformed_df, 2).to_frame().reset_index()
    
    # create a list to sort bar chart by
    if transformed_df[column_name].dtypes == 'category':
        sort_range = list(transformed_df[column_name].cat.categories)
    else: sort_range = list(transformed_df[column_name])

    # create bar chart
    chart = alt.Chart(transformed_df, title=title).mark_bar(color=color).encode(
        x=alt.X('proportion:Q').scale(domain=scale_domain).axis(title=None, labels=False), 
        y=alt.Y(column_name).axis(title=None, labelFontSize=9, labelLimit=180).sort(sort_range),
        text=alt.Text('proportion').format('.0%')            
    ).properties(height=chart_height)
    # create bar labels
    text = chart.mark_text(align='left', dx=2, color=text_color)
    chart = chart + text

    return chart

# selecting bar(s) to color differently
def bar_conditional_color(barchart:alt, value_list:list, field:str, color:str = secondary_color):
    bar_color = barchart.transform_filter(
        alt.FieldOneOfPredicate(field=field, oneOf=value_list)
        ).encode(color=alt.value(color))
    return bar_color

# pie chart
@st.cache_data(ttl=1*60*60)
def pie_chart(df:pd.DataFrame, column_name:str, colors:list, outer_pie_radius:int=50, inner_pie_radius:int=20,
              text_radius:int=70):
    # count attribute occurence
    transformed_df = df[column_name].value_counts(normalize=True)
    # round to two decimal places
    transformed_df = round(transformed_df, 2).to_frame().reset_index()
    transformed_df['text_column'] = transformed_df[column_name] + '\n('\
        + (transformed_df['proportion']*100).round(2).astype(str) + '%)'
    
    # Create pie chart
    pie_chart = alt.Chart(transformed_df)\
        .mark_arc(outerRadius=outer_pie_radius, innerRadius=inner_pie_radius).encode(
            alt.Theta("proportion:Q").stack(True),
            alt.Color(column_name, scale=alt.Scale(range=colors)).legend(None)
    )

    text = pie_chart.mark_text(radius=text_radius, size=10).encode(text='text_column')

    return pie_chart + text

def insights_df_transformation(df:pd.DataFrame, header:str, persona_col:str, complete_str:str='All customers'):
    st.markdown(f'''<h4 style='color:{secondary_color};'>{header}</h4>''', unsafe_allow_html=True)
        
    count_df = df[persona_col].value_counts().to_frame().reset_index()
    fig = plt.figure()
    squarify.plot(sizes=count_df['count'], label=count_df[persona_col], alpha=0.7, pad=True,
                  value=count_df['count'], color=[primary_color, text_color, tertiary_color, grey_color])
    plt.axis('off')
    with st.expander(f'Percentage of customers in {header.lower()} segments *(click to expand/collapse)*'):
        col1, col2 = st.columns(2)
        col1.pyplot(fig, use_container_width=True)

    if persona_col == 'channel_persona':
        selection_list = df[persona_col].unique().tolist()
    else:
        selection_list = (df[persona_col].unique() + 's').tolist()
    selection_list.insert(0, complete_str)
    selection = st.radio('Customer segments', options=selection_list, horizontal=True)
    if selection == complete_str: df = df
    else: df = df.loc[df[persona_col]==selection.removesuffix('s'), :]
    return df, selection

# segmentation insights
def display_segmentation_insights():
    st.header('')
    # create columns and containers
    demography_col, segments_col = st.columns([1,1.8])
    with new_container(command=demography_col, height=236):
        st.header('')
        st.markdown('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: rgba(62, 14, 18, 1);transform: ;msFilter:;"><path d="M20.29 8.29 16 12.58l-1.3-1.29-1.41 1.42 2.7 2.7 5.72-5.7zM4 8a3.91 3.91 0 0 0 4 4 3.91 3.91 0 0 0 4-4 3.91 3.91 0 0 0-4-4 3.91 3.91 0 0 0-4 4zm6 0a1.91 1.91 0 0 1-2 2 1.91 1.91 0 0 1-2-2 1.91 1.91 0 0 1 2-2 1.91 1.91 0 0 1 2 2zM4 18a3 3 0 0 1 3-3h2a3 3 0 0 1 3 3v1h2v-1a5 5 0 0 0-5-5H7a5 5 0 0 0-5 5v1h2z"></path></svg>', unsafe_allow_html=True)
        st.markdown(f'Who are the typical {store_name} customers?')
        st.button('Know More', type="primary", key='demography')
    with segments_col:
        col1, col2 = st.columns(2)
        with new_container(command=col1, height=110):
            st.markdown('''<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" style="fill: rgba(62, 14, 18, 1);transform: ;msFilter:;"><path d="M11.707 2.293A.997.997 0 0 0 11 2H6a.997.997 0 0 0-.707.293l-3 3A.996.996 0 0 0 2 6v5c0 .266.105.52.293.707l10 10a.997.997 0 0 0 1.414 0l8-8a.999.999 0 0 0 0-1.414l-10-10zM13 19.586l-9-9V6.414L6.414 4h4.172l9 9L13 19.586z"></path><circle cx="8.353" cy="8.353" r="1.647"></circle></svg>
                        Purchase behaviour''', unsafe_allow_html=True)
            st.button('Know More', type="primary", key='purchase_behaviour')
        with new_container(command=col1, height=110):
            st.markdown('''<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" style="fill: rgba(62, 14, 18, 1);transform: ;msFilter:;"><path d="M19 4H6V2H4v18H3v2h4v-2H6v-5h13a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1zm-1 9H6V6h12v7z"></path></svg>
                        Competitor landscape''', unsafe_allow_html=True)
            st.button('Know More', type="primary", key='competitor_landscape')
        
        with new_container(command=col2, height=110):
            st.markdown('''<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" style="fill: rgba(62, 14, 18, 1);transform: ;msFilter:;"><path d="M21 4c0-1.103-.897-2-2-2H5c-1.103 0-2 .897-2 2v16c0 1.103.897 2 2 2h14c1.103 0 2-.897 2-2V4zM5 4h14v7H5V4zm0 16v-7h14.001v7H5z"></path><path d="M14 7h-4V6H8v3h8V6h-2zm0 8v1h-4v-1H8v3h8v-3z"></path></svg>
                        Preferred products''', unsafe_allow_html=True)
            st.button('Know More', type="primary", key='product_pref')
        with new_container(command=col2, height=110):
            st.markdown('''<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" style="fill: rgba(62, 14, 18, 1);transform: ;msFilter:;"><path d="M5 22h14c1.103 0 2-.897 2-2V9a1 1 0 0 0-1-1h-3V7c0-2.757-2.243-5-5-5S7 4.243 7 7v1H4a1 1 0 0 0-1 1v11c0 1.103.897 2 2 2zM9 7c0-1.654 1.346-3 3-3s3 1.346 3 3v1H9V7zm-4 3h2v2h2v-2h6v2h2v-2h2l.002 10H5V10z"></path></svg>
                        Preferred channels''', unsafe_allow_html=True)
            st.button('Know More', type="primary", key='channel_pref')
    
    st.divider()

    if st.session_state['demography']:
        st.switch_page(page='pages/5_🧍_Typical_Customer.py')

    elif st.session_state['purchase_behaviour']:
        st.switch_page(page='pages/6_🛍️_Purchase_Habits.py')
    
    elif st.session_state['competitor_landscape']:
        st.switch_page(page='pages/7_🚩_Loyalty.py')
    
    elif st.session_state['product_pref']:
        st.switch_page(page='pages/8_🗄️_Products.py')
    
    elif st.session_state['channel_pref']:
        st.switch_page(page='pages/9_🛒_Shopping_Channels.py')

# customer personas
def customer_personas(df:pd.DataFrame):
    # rearrange dataframe
    persona_df = df.sample(100, random_state=2024)
    # persona data transformation
    persona_df['highest_frequency'] = persona_df['highest_frequency'].map(num_freq_groups)
    persona_df['app_conversion'] = persona_df['app_conversion'].map(app_conversion)
    persona_df['id'] = persona_df.index + ": " + persona_df['customer_name']
    
    # set customer profile variables
    persona_columns = list(persona_df.columns)
    customer_name_list = persona_df['id'].tolist()
    customer = st.selectbox('Select a customer whose profile to view', options=customer_name_list)
    persona_df = persona_df[persona_df['id']==customer]

    customer_name = persona_df.iloc[0]['customer_name']
    gender = persona_df.iloc[0]['gender'].lower()
    age = persona_df.iloc[0]['age']
    age_category = persona_df.iloc[0]['age_category']
    shop_for = persona_df.iloc[0]['who_you_usually_shop_for']
    income = persona_df.iloc[0]['income_category']
    saletype = persona_df.iloc[0][f'price_of_items_you_buy_on_each_{store_name.lower()}_trip']
    benefits = persona_df.loc[:, [col for col in persona_columns if 'do_you_use' in col]]
    benefits = pd.melt(benefits)
    benefits = benefits.loc[benefits['value'].notna(), 'value'].tolist()
    benefits = ', '.join(benefits)
    benefits = benefits.replace('bnpl', 'Buy Now, Pay Later')
    if not benefits: benefits = 'None'
    competitors = persona_df.loc[:, [col for col in persona_columns if 'where_else_do_you_shop' in col]]
    competitors = pd.melt(competitors)
    competitors = competitors.loc[competitors['value'].notna(), 'value'].tolist()
    competitors = ', '.join(competitors)
    spend_competitor = persona_df.iloc[0]['competitor_spend_category']
    freq = persona_df.iloc[0]['highest_frequency']
    app_shop = persona_df.iloc[0]['app_conversion']
    num_competitor = persona_df.iloc[0]['number_of_competitors']
    if num_competitor==1: competitor_text='competitor'
    else: competitor_text='competitors'
    emporium_spend = persona_df.iloc[0][f'{store_name.lower()}_spend_category']
    purchase_behaviour = persona_df.iloc[0]['behavioural_persona']
    loyalty = persona_df.iloc[0]['competitor_landscape_persona']
    product_preference = persona_df.iloc[0]['product_persona']
    channel_preference = persona_df.iloc[0]['channel_persona']
    behaviour_rec = persona_df.iloc[0]['behavioural_recommendation']
    loyalty_rec = persona_df.iloc[0]['competitor_landscape_recommendation']
    product_pref_rec = persona_df.iloc[0]['product_pref_recommendation']
    channel_pref_rec = persona_df.iloc[0]['channel_pref_recommendation']
    
    # USER INTERFACE
    # create header with customer id
    st.markdown(f"""
                <div style='background-color: {secondary_color}; border-radius:2px; padding-left: 10px; color: white; font-size: 36px; font-weight: bold;'>{customer_name}</div>
                """, unsafe_allow_html=True)
        
    # set three columns
    col1, col2, col3 = st.columns(3, gap='medium')
    
    #COLUMN 1.......................................
    with col1:
        # choose image according to age and gender
        if age_category == 'Under 25':
            image = f'pictures/under_25_{gender}.png'
        elif age_category == '25 to 34':
            image = f'pictures/25_35_{gender}.png'
        elif age_category == '35 to 44':
            image = f'pictures/35_45_{gender}.png'
        elif age_category == '45 to 54':
            image = f'pictures/45_55_{gender}.png'
        elif age_category == '55 to 69':
            image = f'pictures/55_70_{gender}.png'
        else:
            image = f'pictures/over_70_{gender}.png'
        if gender == 'female':pronoun = 'She'
        else: pronoun= 'He'

        # write the bio and purchasing behaviours
        st.markdown(f"""
                    <div style="font-size:14px; padding-top:15px; padding-bottom:15px;">
                        <div style="color:{secondary_color};"><strong>Bio</strong></div>
                        <div style='display: flex; justify-content: space-between; font-size: 14px; padding-top: 2px;'>
                            <div style='text-align: left; font-weight: bold;'>Age: </div>
                            <div style='text-align: right;'>{age} years old</div>
                        </div>
                        <div style='display: flex; justify-content: space-between; font-size: 14px;'>
                            <div style='text-align: left; font-weight: bold;'>Income range: </div>
                            <div style='text-align: right;'>{income}</div>
                        </div>
                        <div style='display: flex; justify-content: space-between; font-size: 14px;'>
                            <div style='text-align: left; font-weight: bold;'>Spend range: </div>
                            <div style='text-align: right;'>{emporium_spend}</div>   
                        </div>
                        <div style='display: flex; justify-content: space-between; font-size: 14px;'>
                            <div style='text-align: left; font-weight: bold;'>Financial services: </div>
                            <div style='text-align: right;'>{benefits}</div> 
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # write the image
        st.image(image, use_column_width='always')
        
    # COLUMN 2.......................................
    with col2:
        st.markdown(f"""
                    <div style="font-size:14px; padding-top:15px; padding-bottom:15px;">
                        <div style="color:{secondary_color};"><strong>Purchasing habits</strong></div>
                        <div>
                            {customer_name} shops from {store_name} <strong>{str(freq).lower()}</strong>.
                            {pronoun} says and I quote, "I usually shop for <strong>{shop_for}</strong> and buy <strong>{saletype.lower()}</strong> items."
                        </div>
                        <div style="padding-top: 2px;">
                            {pronoun} belongs to the <strong>{purchase_behaviour}s</strong>. {behaviour_rec}.
                        </div>
                        <div style="color:{secondary_color}; padding-top: 10px;"><strong>Preferred shopping channels</strong></div>
                        <div>
                            {pronoun} is part of the <strong>{channel_preference}</strong>. {channel_pref_rec}.<br>
                            {pronoun} also {app_shop}.
                        </div>
                        <div style="color:{secondary_color}; padding-top: 10px;"><strong>Preferred products</strong></div>
                        <div>
                            {pronoun} is one of the <strong>{product_preference}s</strong>. {product_pref_rec}.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col3:
        # check if there are listed competitors
        if num_competitor > 0:
            competitor_sentence = f'{pronoun} shops at <strong>{num_competitor} listed {competitor_text}</strong> and spends <strong>{spend_competitor}</strong> there.'
        else:
            competitor_sentence = f'{pronoun} spends <strong>{spend_competitor}</strong> at unlisted competitors.'
        
        # write the competitor landscape and preferred products and shopping channels
        st.markdown(f"""
                    <div style="font-size:14px; padding-top:15px; padding-bottom:15px;">
                        <div style="color:{secondary_color};"><strong>Competitor landscape</strong></div>
                        <div>
                            {pronoun} belongs to the <strong>{loyalty}s</strong>. {loyalty_rec}. <br>
                            {competitor_sentence}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with st.expander(f'The listed competitor(s) include:', expanded=True):
            # create a list to append competitor images
            images_list = []

            # split competitors
            for picture in competitors.split(', '):
                # append competitor images to the list
                try:
                    img = f'pictures/competitors/{picture}.png'
                    images_list.append(img)
                except: pass
            
            # create two columns and split image per column
            num_columns = 2
            num_images_per_column = -(len(images_list) // -num_columns)
            if num_images_per_column == 0: num_images_per_column += 1
            columns = st.columns(num_columns, gap='small')
            
            # use a for loop to display each image in its own column
            for i, column in enumerate(columns):
                start_index = i * num_images_per_column
                end_index = (i+1) * num_images_per_column
                for image in images_list[start_index:end_index]:
                    try: column.image(image, use_column_width='auto')
                    except: pass

    st.divider()

# service agent
def customer_service_agent(df:pd.DataFrame):

    sample_col, main_col = st.columns([1,3])

    with new_container(sample_col):
        st.markdown(f'''
                    This conversational AI agent is still in experimental phase. Here are two tested queries
                    that give accurate results.
                    <li>What segments does [email] belong to?<br> <span style='font-size:12px'> where [email]
                    is an available email address in the data (available for download below), for example,
                    ava.jackson@example.com</span></li>
                    <li>What can I recommend to him/her?</li><br>''',
                    unsafe_allow_html=True)
        csv = convert_df(df=df)
        st.download_button(label='Download data', data=csv, type='primary',
                           file_name='Customer Service Agent Data.csv', mime='text/csv')

    with main_col:
        with st.expander('⚠️ **Important information** *(click to collapse/expand)*', expanded=True):
            st.info('''A default OpenAI key has been provided. When it has expired, the agent will be unable to
                    respond to queries. When this happens, please provide your OpenAI key in the textbox below.
                    We value your privacy and clear this key immediately after your session.''')
            openai_api_key = st.text_input("OpenAI API Key", type="password",
                                            label_visibility='collapsed')
        
        if "messages" not in st.session_state or st.button("Clear conversation history"):
            st.session_state["messages"] = [{"role": "assistant", "content": """My name is Iris, your AI assistant
                                            for customer segmentation! Using customer email addresses, I help you
                                            understand your customers better and provide recommendations that help
                                            you provide more personalised service, leading to happier customers
                                            and better outcomes."""}]
        
        if not openai_api_key: openai_api_key = st.secrets['openai_key']
            
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="What segments does ava.jackson@example.com belong to?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                            openai_api_key=openai_api_key, streaming=True)

            pandas_df_agent = create_pandas_dataframe_agent(llm, df, verbose=True,
                                                            agent_type=AgentType.OPENAI_FUNCTIONS)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)


############### DATA TRANSFORMATION FUNCTIONS ####################

# read service agent data
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# data cleaning and transformation
@st.cache_data(ttl=1*60*60)
def data_wrangling(file:str='emporium_segmentation_data.csv'):
    # read file
    customers_df = pd.read_csv(file)
    # change id datatype to string
    customers_df.customer_id = customers_df.customer_id.astype(str)
    # make id the dataframe index
    customers_df.set_index('customer_id', inplace=True)
    # set column to a variable
    age_column = customers_df.age
    
    # group ages into categories
    age_conditions = [age_column < 25, (age_column >= 25) & (age_column < 35), (age_column >= 35)
                    & (age_column < 45), (age_column >= 45) & (age_column < 55), (age_column >= 55)]
    age_groups = ['Under 25', '25 to 34', '35 to 44', '45 to 54', '55 and above']
    new_age_column = np.select(age_conditions, age_groups)
    customers_df.loc[:, 'age_category'] = pd.Categorical(new_age_column, categories=age_groups, ordered=True)
    
    # set column to a variable
    income_column = customers_df.annual_household_income
    # group income ranges into categories
    income_conditions = [income_column < 50000, (income_column >= 50000) & (income_column < 75000), 
                        (income_column >= 75000) & (income_column < 100000), (income_column >= 100000) & 
                        (income_column < 125000), (income_column >= 125000) & (income_column < 150000),
                        income_column >= 150000]
    income_groups = ['Under $50,000', '$50,000 to $74,999', '$75,000 to $99,999', '$100,000 to $124,999',
                    '$125,000 to $149,999', '$150,000 and above']
    new_income_column = np.select(income_conditions, income_groups)
    customers_df.loc[:, 'income_category'] = pd.Categorical(new_income_column, categories=income_groups, ordered=True)
    
    # calculate how often customers have visited Walmart in the past three months
    frequency_columns = [col for col in customers_df.columns if 'how_often' in col]
    frequency_groups = ['Never', 'Once or twice', 'Monthly', 'Every two weeks', 'Weekly', 'More than once per week']
    num_freq_groups = {'Never':0, 'Once or twice':2, 'Monthly':3, 'Every two weeks':6, 'Weekly':12,
                    'More than once per week':36}
    for column in frequency_columns:
        customers_df[column] = pd.Categorical(customers_df[column], categories=frequency_groups, ordered=True)
        customers_df[f'numeric_{column}'] = customers_df[column].replace(num_freq_groups).astype(int)
    customers_df['highest_frequency'] = customers_df[frequency_columns].max(axis=1)
    
    # change highest frequency to numeric
    customers_df.highest_frequency = customers_df.highest_frequency.replace(num_freq_groups).astype(int)
    # create an empty column for app conversion
    customers_df.loc[:, 'app_conversion'] = None
    # set value to not used for those who have not used the app
    customers_df.loc[(customers_df[f'you_have_used_the_{store_name.lower()}_app_to_shop'] == 'No'), 'app_conversion'] = 'Not used'
    
    # set value to converted if a customer prefers to use the app for any product category
    customers_df.loc[(customers_df[f'you_have_used_the_{store_name.lower()}_app_to_shop'] == 'Yes') &
                    ((customers_df['you_prefer_shopping_clothing_shoes_accessories_with_the_app'] == 'App') |
                    (customers_df['you_prefer_shopping_groceries_with_the_app'] == 'App') | 
                    (customers_df['you_prefer_shopping_health_wellness_beauty_with_the_app'] == 'App') |
                    (customers_df['you_prefer_shopping_home_furniture_appliances_with_the_app'] == 'App')), 
                    'app_conversion'] = 'Converted'
    
    # else set value to churned
    customers_df.loc[customers_df['app_conversion'].isna(), 'app_conversion'] = 'Churned'
    # store column in a variable
    emporium_spend_column = customers_df[f'how_much_do_you_spend_on_each_{store_name.lower()}_trip']
    
    # group emporium_spend ranges into categories
    emporium_spend_conditions = [emporium_spend_column < 30, (emporium_spend_column >= 30) & (emporium_spend_column < 50), 
                                (emporium_spend_column >= 50) & (emporium_spend_column < 100), emporium_spend_column >= 100]
    emporium_spend_groups = ['Under $30', '$30 to $49', '$50 to $99', '$100 and above']
    new_emporium_spend_column = np.select(emporium_spend_conditions, emporium_spend_groups)
    customers_df.loc[:, f'{store_name.lower()}_spend_category'] = pd.Categorical(new_emporium_spend_column,
                                                                    categories=emporium_spend_groups, ordered=True)
    
    # list all columns for competitors
    competitor_cols = [column for column in customers_df.columns if 'where_else_do_you_shop' in column]
    # count competitors
    customers_df.loc[:,'number_of_competitors'] = customers_df[competitor_cols].count(axis=1)
    # store column in a variable
    competitor_spend_column = customers_df.how_much_do_you_spend_on_each_trip_to_competitors
    
    # group competitor_spend ranges into categories
    competitor_spend_conditions = [competitor_spend_column < 30, (competitor_spend_column >= 30) & 
                                (competitor_spend_column < 50), (competitor_spend_column >= 50) & 
                                (competitor_spend_column < 100), (competitor_spend_column >= 100) & 
                                (competitor_spend_column < 150), competitor_spend_column >= 150]
    competitor_spend_groups = ['Under $30', '$30 to $49', '$50 to $99', '$100 to $149', '$150 and above']
    new_competitor_spend_column = np.select(competitor_spend_conditions, competitor_spend_groups)
    customers_df.loc[:, 'competitor_spend_category'] = pd.Categorical(new_competitor_spend_column,
                                                                    categories=competitor_spend_groups, ordered=True)
    
    # set column to a variable
    item_prices = customers_df[f'price_of_items_you_buy_on_each_{store_name.lower()}_trip']
    # group into categories
    price_group = ['All discounted', 'Mostly discounted, some full price', 'Mostly full price, some discounted',
                'All full price']
    customers_df[f'price_of_items_you_buy_on_each_{store_name.lower()}_trip'] = pd.Categorical(item_prices, categories=price_group,
                                                                            ordered=True)
    
    # count number of financial services used
    cashback_cols = [col for col in customers_df.columns if 'do_you_use' in col]
    customers_df.loc[:, 'number_of_money_services_used'] = customers_df.loc[:, cashback_cols].fillna('')\
        .map(lambda x:1 if x != '' else 0).sum(axis=1)
    
    return customers_df

# check for skewness and perform log/cubic/sqaure root transformation if the data is skewed
@st.cache_data(ttl=1*60*60)
def skewness_validation(features_df:pd.DataFrame):
    # for each column
    for column_name in features_df:
        # store column in a variable
        column = features_df[column_name]
        # calculate skewness
        skew = column.skew()
        
        # if skewed
        if (skew < -1) or (skew > 1):
            # perform log, cubic, and square root transformations
            transform_cols = {'log':np.log(column), 'cubic':np.cbrt(column), 'sqrt':np.sqrt(column)}
            transform_skew = {'log':abs(np.log(column).skew()), 'cubic':abs(np.cbrt(column).skew()),
                              'sqrt':abs(np.sqrt(column).skew())}
            
            # find the transformation with best skewness value
            minimum_value = min(value for value in transform_skew.values() if value != float("-inf") 
                                and not np.isnan(value))
            best_transform = [key for key in transform_skew if transform_skew[key] == minimum_value][0]
            
            # transform the column using the best transformation
            features_df[column_name] = transform_cols[best_transform]
    
    return features_df

# create a function to standardize columns and plot kmeans residual errors
@st.cache_data(ttl=1*60*60)
def kmeans(df_features:pd.DataFrame, min_range:int=1, max_range:int=11):
    # initialize standard scaler
    scaler = StandardScaler()
    # standardise the dataframe
    scaler.fit(df_features)
    df_normalized = scaler.transform(df_features)
    df_normalized = pd.DataFrame(df_normalized, index=df_features.index, columns=df_features.columns)
    
    # create a list of sum of squared errors (sse)
    sse = {}
    # for the specified range, find sse of the kmeans clustering algorithm
    for k in range(min_range,max_range):
        kmeans= KMeans(n_clusters=k,n_init= 10,max_iter=300,tol=0.0001,random_state=random_seed)
        a= kmeans.fit(df_normalized)
        sse[k] = a.inertia_
    
    return df_normalized, sse

# get clusters, avg feature value, and relative feature importance of kmeans clusters
@st.cache_data(ttl=1*60*60)
def kmeans_cust_seg(df_raw:pd.DataFrame, df_norm:pd.DataFrame, n_clusters:int):
    # perform kmeans algorithm on selected number of clusters
    kmeans = KMeans(n_clusters, random_state=random_seed, n_init=10)
    kmeans.fit(df_norm)
    cluster_labels = kmeans.labels_
    df_norm = df_norm.assign(Segment = cluster_labels)
    
    # melt dataframe for avg feature value per segment
    df_melt = pd.melt(df_norm, id_vars='Segment', var_name='Attribute', value_name='Value')
    
    # calculate feature relative importance
    pop_avg = df_raw.mean()
    df_raw = df_raw.assign(Segment = cluster_labels)
    cluster_avg = df_raw.groupby('Segment').mean()
    relative_imp = cluster_avg/pop_avg - 1
    
    return df_raw, df_norm, df_melt, relative_imp

# engineer features for kmeans segmentation
@st.cache_data(ttl=1*60*60)
def kmeans_feature_eng(df:pd.DataFrame, selected_cols:list, cat_col:bool, cat_col_name:str):
    # select the columns
    persona_df = df.loc[:, selected_cols]
    # numerically code categorical columns
    if cat_col:
        persona_df.loc[:, cat_col_name] = df[cat_col_name].cat.codes
    
    # create a copy of the dataframe for the model
    features_df = persona_df.copy()
    # check for skewness and perform log/cubic/square root transformation if the data is skewed
    features_df = skewness_validation(features_df=features_df)
    
    return persona_df, features_df

# build kmeans segments
@st.cache_data(ttl=1*60*60)
def kmeans_segments(df:pd.DataFrame, selected_cols:list, cat_col:bool, cat_col_name:str, n_clusters:int,
                    segment_type:str, segment_personas:dict):
    persona_df, features_df = kmeans_feature_eng(df, selected_cols, cat_col, cat_col_name)
    # use elbow spree to check sse
    df_normalized, seg_sse = kmeans(df_features=features_df)

    # inspect feature values and importance across segments
    persona_df, df_normalized, df_melt, relative_imp_df = kmeans_cust_seg(df_raw=persona_df, df_norm=df_normalized,
                                                                      n_clusters=n_clusters)
    
    # rename cols
    persona_df.rename(columns={'Segment':f'{segment_type}_segment'}, inplace=True)
    # create persona column with each segment
    persona_df[f'{segment_type}_persona'] = persona_df[f'{segment_type}_segment'].map(segment_personas)
    
    return persona_df, seg_sse, df_normalized, df_melt, relative_imp_df

# create a function to plot kmodes residual errors
@st.cache_data(ttl=1*60*60)
def kmodes(df:pd.DataFrame, min_range:int=1, max_range:int=11):
    # create list for sses
    sse = {}

    # for specified range, run Kmodes algorithm
    for k in range(min_range,max_range):
        kmodes= KModes(n_clusters=k,init='Huang', n_init=10, random_state=random_seed)
        clusters= kmodes.fit(df)
        sse[k] = clusters.cost_
    
    return sse

# get clusters
@st.cache_data(ttl=1*60*60)
def kmodes_cust_seg(df:pd.DataFrame, n_clusters:int):
    # kmodes algorithm for specified number of clusters
    kmodes= KModes(n_clusters=n_clusters, init='Huang', n_init=10, random_state=random_seed)
    kmodes.fit(df)
    labels = kmodes.predict(df)
    results = pd.DataFrame({'Segment': labels}, index=df.index)
    
    # join results with original dataframe
    df = pd.concat([results, df], axis=1)
    
    return df

# build kmodes segments
@st.cache_data(ttl=1*60*60)
def kmodes_segments(df:pd.DataFrame, selected_cols:list, n_clusters:int, segment_type:str, segment_personas:dict):
    # create channel preference df
    persona_df = df.loc[:, selected_cols].fillna(0).map(lambda x:1 if x!=0 else 0)
    features_df = persona_df.copy()

    # plot sses
    seg_sse = kmodes(df=features_df)
    # create channel segments
    persona_df = kmodes_cust_seg(df=persona_df, n_clusters=n_clusters)
    # rename cols
    persona_df.rename(columns={'Segment':f'{segment_type}_segment'}, inplace=True)
    # create persona column with each segment
    persona_df[f'{segment_type}_persona'] = persona_df[f'{segment_type}_segment'].map(segment_personas)
    
    return persona_df, seg_sse

# build segmentation model
@st.cache_data(ttl=1*60*60)
def segmentation_model_results(df:pd.DataFrame, behaviour_selected_cols:list, behaviour_cat_col:bool,
                               behaviour_cat_col_name:str, behaviour_clusters:int, behaviour_segment_type:str,
                               behaviour_segment_personas:dict, loyalty_selected_cols:list, loyalty_cat_col:bool,
                               loyalty_cat_col_name:str, loyalty_clusters:int, loyalty_segment_type:str,
                               loyalty_segment_personas:dict, product_selected_cols:list, product_cat_col:bool,
                               product_cat_col_name:str, product_clusters:int, product_segment_type:str,
                               product_segment_personas:dict, channel_selected_cols:list, channel_clusters:int,
                               channel_segment_type:str, channel_segment_personas:dict):
    # build purchase behaviour segments
    behaviour_seg_df, behaviour_seg_sse, behaviour_df_normalized, behaviour_df_melt, behaviour_relative_imp_df = \
        kmeans_segments(df=df, selected_cols=behaviour_selected_cols, cat_col=behaviour_cat_col,
                        cat_col_name=behaviour_cat_col_name, n_clusters=behaviour_clusters,
                        segment_type=behaviour_segment_type, segment_personas=behaviour_segment_personas)
    
    # build loyalty/competitor landscape segments
    loyalty_seg_df, loyalty_seg_sse, loyalty_df_normalized, loyalty_df_melt, loyalty_relative_imp_df = \
        kmeans_segments(df=df, selected_cols=loyalty_selected_cols, cat_col=loyalty_cat_col,
                        cat_col_name=loyalty_cat_col_name, n_clusters=loyalty_clusters,
                        segment_type=loyalty_segment_type, segment_personas=loyalty_segment_personas)
    
    # build product preference segments
    product_seg_df, product_seg_sse, product_df_normalized, product_df_melt, product_relative_imp_df = \
        kmeans_segments(df=df, selected_cols=product_selected_cols, cat_col=product_cat_col,
                        cat_col_name=product_cat_col_name, n_clusters=product_clusters,
                        segment_type=product_segment_type, segment_personas=product_segment_personas)
    
    # build channel preference segments
    channel_seg_df, channel_seg_sse = \
        kmodes_segments(df=df, selected_cols=channel_selected_cols, n_clusters=channel_clusters,
                        segment_type=channel_segment_type, segment_personas=channel_segment_personas)
    
    # concat all segmentation results
    seg_results = pd.concat([
        behaviour_seg_df.loc[:, ['behavioural_segment', 'behavioural_persona']],
        loyalty_seg_df.loc[:, ['competitor_landscape_segment', 'competitor_landscape_persona']],
        product_seg_df.loc[:, ['product_segment', 'product_persona']],
        channel_seg_df.loc[:, ['channel_segment', 'channel_persona']]
    ], axis=1, ignore_index=False)

    # include segments meanings
    seg_results['behavioural_description'] = seg_results['behavioural_persona'].apply(lambda x: purchase_beh_segment[f'{x}s'][0])
    seg_results['behavioural_recommendation'] = seg_results['behavioural_persona'].apply(lambda x: purchase_beh_segment[f'{x}s'][1])
    seg_results['competitor_landscape_description'] = seg_results['competitor_landscape_persona'].apply(lambda x: loyalty_segment[f'{x}s'][0])
    seg_results['competitor_landscape_recommendation'] = seg_results['competitor_landscape_persona'].apply(lambda x: loyalty_segment[f'{x}s'][1])
    seg_results['product_pref_description'] = seg_results['product_persona'].apply(lambda x: product_pref_segment[f'{x}s'][0])
    seg_results['product_pref_recommendation'] = seg_results['product_persona'].apply(lambda x: product_pref_segment[f'{x}s'][1])
    seg_results['channel_pref_description'] = seg_results['channel_persona'].apply(lambda x: channel_pref_segment[x][0])
    seg_results['channel_pref_recommendation'] = seg_results['channel_persona'].apply(lambda x: channel_pref_segment[x][1].strip())
    
    return behaviour_seg_sse, behaviour_df_normalized, behaviour_df_melt, behaviour_relative_imp_df,\
        loyalty_seg_sse, loyalty_df_normalized, loyalty_df_melt, loyalty_relative_imp_df, product_seg_sse,\
            product_df_normalized, product_df_melt, product_relative_imp_df, channel_seg_df, channel_seg_sse, seg_results

@st.cache_data(ttl="1h")
def convert_df(df:pd.DataFrame):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


############### READING DATA ####################

# load and clean data
customers_df = data_wrangling()

# segmentation model variables
behaviour_selected_cols = ['highest_frequency','annual_household_income',
                           f'how_much_do_you_spend_on_each_{store_name.lower()}_trip', 'number_of_money_services_used']
behaviour_cat_col_name = f'price_of_items_you_buy_on_each_{store_name.lower()}_trip'
behaviour_segment_personas = {0:'Value Seeker', 1:'Luxury Seeker', 2:'Savvy Spender', 3:'Bargain Hunter'}
loyalty_selected_cols = ['highest_frequency', f'how_much_do_you_spend_on_each_{store_name.lower()}_trip',
                         'number_of_competitors', 'how_much_do_you_spend_on_each_trip_to_competitors']
loyalty_segment_personas = {0:'Wanderer', 1:'Explorer', 2:'Loyalist'}
product_selected_cols = [col for col in customers_df.columns if 'numeric_how_often' in col]
product_segment_personas = {0:'Category Hopper', 1:'Pantry Planner', 2:'Style & Sustenance Seeker',
                            3:'Universal Shopper'}
channel_selected_cols = [col for col in customers_df.columns if 'you_prefer_shopping' in col]
channel_segment_personas = {0:'Digital Crew', 1:'Click & Brick Crew', 2:'Omnichannel Crew',
                            3:'Brick & Mortar Crew'}

# build segmentation model
behaviour_seg_sse, behaviour_df_normalized, behaviour_df_melt, behaviour_relative_imp_df,\
    loyalty_seg_sse, loyalty_df_normalized, loyalty_df_melt, loyalty_relative_imp_df, product_seg_sse,\
        product_df_normalized, product_df_melt, product_relative_imp_df, channel_seg_df, channel_seg_sse, seg_results = \
            segmentation_model_results(df=customers_df, behaviour_selected_cols=behaviour_selected_cols,
                                       behaviour_cat_col=True, behaviour_cat_col_name=behaviour_cat_col_name,
                                       behaviour_clusters=4, behaviour_segment_type='behavioural',
                                       behaviour_segment_personas=behaviour_segment_personas,
                                       loyalty_selected_cols=loyalty_selected_cols, loyalty_cat_col=False,
                                       loyalty_cat_col_name='', loyalty_clusters=3,
                                       loyalty_segment_type='competitor_landscape',
                                       loyalty_segment_personas=loyalty_segment_personas,
                                       product_selected_cols=product_selected_cols, product_cat_col=False,
                                       product_cat_col_name='', product_clusters=4, product_segment_type='product',
                                       product_segment_personas=product_segment_personas,
                                       channel_selected_cols=channel_selected_cols, channel_clusters=4,
                                       channel_segment_type='channel',
                                       channel_segment_personas=channel_segment_personas)

# join customer df and seg model df together
complete_model_df = pd.concat([customers_df, seg_results], axis=1, ignore_index=False)

agent_df = pd.concat([complete_model_df.loc[:, ['email_address']],
                      complete_model_df.iloc[:, 53:]],
                      axis=1, ignore_index=False)
agent_df = agent_df.loc[:, [col for col in agent_df.columns if 'description' not in col
                            and 'segment' not in col]]
agent_df.rename(columns={'behavioural_persona':'behavioural_segment', 'competitor_landscape_persona':
                         'competitor_landscape_segment', 'product_persona':'preferred_product_category_segment',
                         'channel_persona':'preferred_shopping_channel_segment', 'product_pref_recommendation':
                         'preferred_product_category_recommendation', 'channel_pref_recommendation':
                         'preferred_shopping_channel_recommendation'}, inplace=True)