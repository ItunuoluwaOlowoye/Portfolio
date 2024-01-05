import numpy as np
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
import random
import numpy as np
import base64
import streamlit as st
from PIL import Image
from django.contrib.auth import authenticate

# set random seed for KMeans and KModes reproducibility
random.seed(1)

def password_entered(): # storing session usernames and passwords
    #Checks whether a password entered by the user is correct.
    user = authenticate(username=st.session_state['username'], password=st.session_state['password'])
    if user is not None: # if username is inputed
        st.session_state['password_correct'] = True # initialize storing session password
        st.session_state.user = user # add the user object to Streamlit session
    else:
        st.session_state['password_correct'] = False # don't initialize storing session password

def authenticate_user(placeholder, sb_placeholder): # authenticate users with Django
    if 'password_correct' not in st.session_state: # first run, session password not initialized
        st.text_input('Username (this is case sensitive)', on_change=password_entered, key='username')
        st.text_input('Password', type='password', on_change=password_entered, key='password') # show inputs for username + password.
        login = st.button('Log in') # add log in button
        if login: return False # don't log in, instead save the session user credentials
        
    elif not st.session_state['password_correct']: # Password not correct, show input + error.
        st.text_input('Username (this is case sensitive)', on_change=password_entered, key='username')
        st.text_input('Password', type='password', on_change=password_entered, key='password') # show inputs for username + password.
        login = st.button('Log in') # add log in button
        if login:
            st.error('❗ User not known or password incorrect')
            return False
        
    else: # Password correct
        placeholder.empty(); sb_placeholder.empty() # clear placeholders
        return True

def page_intro(header,body): # default page elements
    logo = Image.open('misc/pictures/browser-tab-logo.png')
    #inset_logo = resize_image(logo,30)
    logo_column, header_column = st.columns([1,25]) # create columns for logo and header; ratio needs adjustment if layout is changed to centered
    logo_column.title('')
    logo_column.image(logo) # insert the body logo
    header_column.title(header) # write the header
    st.write(body) # write the body
    
    with st.sidebar: # add the creed to the sidebar
        sb_placeholder = st.empty() # add to a container
        with sb_placeholder.container():
            st.title('About')
            st.markdown('''This is a segmentation model for retail customers''', unsafe_allow_html=True)
    return sb_placeholder # store this sidebar placeholder to a variable when calling the function

def page_header(header, image_path):
    # create a style for inserting the image
    st.markdown("""
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:bold !important;
        font-size:32px !important;
    }
    .logo-img {
        float:right;
    }
    </style>
    """, unsafe_allow_html=True)
    # insert logo image and header
    st.markdown(f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}"
            style="max-width: 10%; max-height: 10%; object-fit:contain;">
        </div><font class="logo-text">{header}</font>
        """,
        unsafe_allow_html=True)

# caches are used to store results for a defined number of seconds

# load, clean, and transform data
@st.cache_data(ttl=10*60)
def data_transform_phase_one(filepath):
    df = pd.read_csv(filepath)
    # if there is an unnamed index column, remove from the dataframe
    for column in df.columns:
        if ('Unnamed' in column) or ('lower_bound' in column) or ('upper_bound') in column:
            df.drop(column, axis=1, inplace=True)
    
    # replace _NA, _Response and _ with empty strings in the column names
    df.columns = df.columns.str.replace('_NA','').str.replace('_Response','')\
    .str.replace('_','').str.replace(' & ','&').str.replace(' - ','-').str.replace(r'(:)$','',regex=True)\
    .str.replace('- ','-')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('_NA','').str.replace('_Response','')\
                .str.replace('_','').str.replace(' & ','&').str.replace(' - ','-').str.replace(r'(:)$','',regex=True)\
                    .str.replace('- ','-')
    
    # convert Respondent ID column from integer to string
    # if the id was initially written as exponential, we need to remove the .0 at the end after converting to a string
    df['Respondent ID'] = df['Respondent ID'].astype(str).str.replace('.0','',regex=False)
    
    # fill null values with prefer not to say
    df['Gender'].fillna('prefer not to say', inplace=True)
    
    # convert the age ranges to categories from youngest to oldest
    age_category = ['19-25', '26-35', '36-45', '46-55', '56-70', '71+']
    df['How old are you?'] = pd.Categorical(df['How old are you?'], categories=age_category, ordered=True)
    
    # create an age column with the lower bound of the age ranges
    split_age = df['How old are you?'].str.split('-',expand=True)
    df['age'] = split_age[0].str.extract('(\d+)').astype(int)
    df_nullincome = pd.read_csv('wwsurvey.csv')
    df_nullincome = df_nullincome[['What is your total annual household income?_Response']]
    df['What is your total annual household income? Untreated'] = df_nullincome
    
    # create an income column with the lower and upper bounds of the income ranges
    split_income = df['What is your total annual household income?'].str.replace(',','')
    lower_income_range = split_income.str.split().str[0].str.extract('(\d+)').astype(float)
    min_lower_income_range = lower_income_range.min()
    lower_income_range.fillna(min_lower_income_range/2, inplace=True)
    higher_income_range = split_income.str.split().str[-1].str.extract('(\d+)').astype(float)
    max_higher_income_range = higher_income_range.max()
    higher_income_range.fillna(max_higher_income_range*2, inplace=True)
    df['income'] = (lower_income_range + higher_income_range)/2
    total_income = df['income']
    income_condition = [total_income<120000, total_income<240000, total_income<360000, total_income<480000, total_income<600000,
                        total_income<1200000, total_income>=1200000]
    income_category = ['Less than R120000', 'R120000 to R239000', 'R240000 to R359000', 'R360000 to R479000', 'R480000 to R599000',
                        'R600000 to R1200000', 'R1200000 or more']
    df['What is your total annual household income?'] = np.select(
        income_condition, income_category, default=df['What is your total annual household income?']
    )
    
    # convert the income ranges to categories from smallest to largest
    df['What is your total annual household income?'] = pd.Categorical(df['What is your total annual household income?'],
                                                                                    categories=income_category, ordered=True)
    
    # select columns with the above prefix
    freq_cols = []
    for column in df.columns:
        if 'Thinking back over the last 3 months, how regularly did you shop with Woolworths for different products' in column:
            freq_cols.append(column)
    freq_cat_list = ['Never', 'Once or twice', 'Monthly', 'Every 2 weeks', 'Weekly', 'More than once per week']
    freq_num_list = [0,2,3,6,12,52]
    freq_num_replacement = dict(zip(freq_cat_list, freq_num_list))
    freq_cat_replacement = dict(zip(freq_num_list, freq_cat_list))
    
    for column in freq_cols:
        # fill nulls in these columns with 'Never'
        df[column].fillna('Never',inplace=True)
        # turn the frequencies into categorical variables
        df[column] = pd.Categorical(df[column], ordered=True,
                                                    categories=freq_cat_list)
        # create a numerical frequency column
        product_type = column.split(':')[1]
        df['Shopping Frequency:'+product_type] = df[column].replace(freq_num_replacement)
    
    customers_columns = list(df.columns)
    df['general_freq'] = df.loc[:, [col for col in customers_columns
                                                                    if 'Shopping Frequency' in col]].max(axis=1)
    df['Thinking back over the last 3 months, how regularly did you shop with Woolworths']\
    = df['general_freq'].replace(freq_cat_replacement)
    df['Thinking back over the last 3 months, how regularly did you shop with Woolworths'] = \
        pd.Categorical(df['Thinking back over the last 3 months, how regularly did you shop with Woolworths'], 
        ordered=True, categories=freq_cat_list)
    
    # select columns for how much they spend at Woolworths
    spend_cols = []
    for column in df.columns:
        if 'How much do you normally spend, per shop, when you shop at Woolworths?' in column:
            spend_cols.append(column)
    spend_cat_list = ['R350-R500', 'R500-R1000', 'R1000-R2000', 'Above R2000']
    for column in spend_cols:
        # turn the spend ranges into categorical variables
        df[column] = pd.Categorical(df[column], ordered=True,
                                                    categories=spend_cat_list)
        # create a numerical spend column
        lower_spend_range = df[column].str.split('-').str[0].str.extract('(\d+)').astype(float)
        higher_spend_range = df[column].str.split('-').str[-1].str.extract('(\d+)').astype(float)
        price = column.split('?')[1]
        df['Spend at Woolworths:'+price] = (lower_spend_range + higher_spend_range)/2
    
    # select columns for whether they have used the app
    app_cols = []
    for column in df.columns:
        if 'Have you used the Woolworths app to shop?' in column:
            app_cols.append(column)
    
    # Coalesce answers
    df['Have you used the Woolworths app to shop?'] = df[
        app_cols
    ].bfill(axis=1).iloc[:,0]
    
    # drop uncoalesced source columns
    df.drop(app_cols,axis=1,inplace=True)
    
    # select columns with competitors
    competitor_cols = []
    for column in df.columns:
        if 'Where else do you shop for food and clothing online, if not Woolworths?' in column:
            competitor_cols.append(column)
    
    # create a column with number of competitors each customer frequents
    df['Number of competitors'] = df[competitor_cols].count(axis=1)
    competitor_spend = df['How much do you spend at these places each month?Open-Ended Response']
    
    # fill null values with zero
    df['How much do you spend at these places each month?Open-Ended Response'].fillna('No value',inplace=True)
    lower_comp_spend_range = competitor_spend.str.split('-').str[0].str.extract('(\d+)').astype(float)
    higher_comp_spend_range = competitor_spend.str.split('-').str[-1].str.extract('(\d+)').astype(float)
    df['competitor_spend'] = (lower_comp_spend_range + higher_comp_spend_range)/2
    competitor_spend = df['competitor_spend']
    comp_condition = [(competitor_spend<500)&(competitor_spend>=350), competitor_spend<1000, competitor_spend<2000, competitor_spend>=2000]
    comp_choice = spend_cat_list
    df['How much do you spend at these places each month?Open-Ended Response'] = np.select(
        comp_condition, comp_choice, default='No value'
    )
    
    # convert the spend ranges to categories from smallest to largest
    df['How much do you spend at these places each month?Open-Ended Response'] = \
        pd.Categorical(df['How much do you spend at these places each month?Open-Ended Response'],
                        categories=['No value']+spend_cat_list, ordered=True)
    
    # select columns for fullprice or discount
    sale_benefits_cols = []
    for column in df.columns:
        if 'How would you describe the fashion/ home/beauty items you usually buy?' in column:
            sale_benefits_cols.append(column)
    sale_categories=['All sale / discount items', 'Mostly sale / discount items, with some full price items',
    'Mostly full price, with some sale / discount items', 'All full price items']
    for column in sale_benefits_cols:
        # turn the sale types into categorical variables
        df[column] = pd.Categorical(df[column], ordered=True,
                                                    categories=sale_categories)
    return df

@st.cache_data(ttl=10*60)
def data_transform_phase_two(df):
    # create a list of all columns
    cust_cols = list(df.columns)
    
    # list columns that don't need to be unpivoted
    feature_cols = ['Respondent ID', 'Gender', 'How old are you?', 'Who do you normally shop for?',
    'What is your total annual household income?','How much do you spend at these places each month?Open-Ended Response', 'age',
    'income', 'Have you used the Woolworths app to shop?', 'Number of competitors'] +\
    [col for col in cust_cols if 'Thinking back over the last 3 months, how regularly did you shop with Woolworths' in col]
    
    # create a dataframe with the columns that do not need to be unpivoted
    features = df[feature_cols]
    
    # list columns with basic customer features and equal number of items to be unpivoted
    equal_unpivot_cols = ['Respondent ID', 'Gender', 'How old are you?', 'age', 'What is your total annual household income?'] +\
    [col for col in cust_cols if 'Over the last 3 months, which method best describes how you most often shop at Woolworths' in col] +\
        [col for col in cust_cols if 'How much do you normally spend, per shop, when you shop at Woolworths' in col] +\
            [col for col in cust_cols if 'How would you describe the fashion/ home/beauty items you usually buy?' in col] +\
                [col for col in cust_cols if 'Spend at Woolworths' in col]
    
    # create a dataframe with the columns with equal number of items to be unpivoted
    unpivot_features = df[equal_unpivot_cols]
    
    # put each product's channel preference in a list
    food_channel_cols, fashion_channel_cols, beauty_channel_cols, home_channel_cols = ([] for _ in range(4))
    channel_pref = 'Over the last 3 months, which method best describes how you most often shop at Woolworths?'
    for column in unpivot_features.columns:
        if channel_pref in column:
            if 'Food' in column:
                food_channel_cols.append(column)
            elif 'Fashion' in column:
                fashion_channel_cols.append(column)
            elif 'Home' in column:
                home_channel_cols.append(column)
            else:
                beauty_channel_cols.append(column)
    
    # put each spend value column in a list (category and numeric)
    spend_cat_cols, spend_num_cols = ([] for _ in range(2))
    spend_category = 'How much do you normally spend, per shop, when you shop at Woolworths?'
    spend_numeric = 'Spend at Woolworths'
    for column in unpivot_features.columns:
        if spend_category in column:
            spend_cat_cols.append(column)
        elif spend_numeric in column:
            spend_num_cols.append(column)
    
    # put each sale type column in a list 
    sale_type_cols = []
    sale_type = 'How would you describe the fashion/ home/beauty items you usually buy?'
    for column in unpivot_features.columns:
        if sale_type in column:
            sale_type_cols.append(column)
    
    # unpivot the dataframe
    unpivot_features = pd.lreshape(unpivot_features, {channel_pref+':Food': food_channel_cols, channel_pref+':Fashion':fashion_channel_cols,
                                                    channel_pref+':Beauty&Health':beauty_channel_cols, channel_pref+':Home':home_channel_cols,
                                                    spend_category:spend_cat_cols, spend_numeric:spend_num_cols,
                                                    sale_type:sale_type_cols}, dropna=False)
    
    # list columns with basic customer features and benefits features
    benefit_features_cols = ['Respondent ID', 'Gender', 'How old are you?', 'age', 'What is your total annual household income?'] +\
        [col for col in cust_cols if 'Do you have any of the following? Please tick all the apply' in col]
    
    # create a dataframe with benefits features
    benefit_features = df[benefit_features_cols]
    
    # put each benefits column in a list 
    benefit_features_cols = []
    benefits = 'Do you have any of the following? Please tick all the apply'
    for column in benefit_features.columns:
        if benefits in column:
            benefit_features_cols.append(column)
    
    # unpivot the dataframe
    benefit_features = pd.lreshape(benefit_features, {benefits:benefit_features_cols}, dropna=False)
    
    # select columns with the above prefix
    competitor_cols = []
    for column in df.columns:
        if 'Where else do you shop for food and clothing online, if not Woolworths?' in column:
            competitor_cols.append(column)
    
    # list columns with basic customer features and competitor features
    competitor_df_cols = ['Respondent ID', 'Gender', 'How old are you?', 'What is your total annual household income?'] + competitor_cols
    
    # create a competitors dataframe
    competitor_features = df[competitor_df_cols]
    
    # unpivot the dataframe
    competitor_features = pd.lreshape(competitor_features, {'Where else do you shop for food and clothing online, if not Woolworths?':
                                                        competitor_cols}, dropna=False)
    return features, unpivot_features, benefit_features, competitor_features

@st.cache_data(ttl=10*60)
def dashboard_bar_charts(df,column,title,type='univariate',dformat='normal',x_labels=False,x_title=None,id='Respondent ID',limit=False, limit_number=5, hide_record='',additional_text=''):
    # if univariate, to count the number of occurences for each unique value and sort in descending order
    if type == 'univariate':
        chart_df = df[column].value_counts().to_frame().reset_index()\
            .rename(columns={column:'count','index':column})
        # to limit to top n rows
        if limit == True:
            chart_df = chart_df.iloc[:limit_number]
    
    # if bivariate, group by feature and calculate median, filling null values with zero
    else:
        chart_df = df.groupby(column).median().reset_index().fillna(0)
    
    # to select the first column
    num_col = chart_df.columns[1]
    
    # to calculate proportions
    if dformat != 'normal':
        chart_df['percent_val'] = (chart_df[num_col]/len(df[id].unique())*100).astype(int)
    else: 
        chart_df['percent_val'] = (chart_df[num_col]/chart_df[num_col].sum()*100).astype(int)
    chart_df['percent_val_text'] = chart_df['percent_val'].astype(str) + '%'
    
    # to hide any specified values
    chart_df = chart_df[chart_df[column] != hide_record]
    
    # to sort the chart
    if df[column].dtypes == 'category': sort_range = list(df[column].dtypes.categories)
    else: sort_range = list(chart_df[column])
    
    # to choose x and y axis values
    if type == 'univariate': 
        chart_text = 'percent_val_text'
        x = alt.X(num_col, axis=alt.Axis(labels=x_labels, title=x_title))
        y = alt.Y(column, axis=alt.Axis(title=None, labelFontSize=9), sort=sort_range)
    else: 
        chart_text = None
        y = alt.Y(num_col, axis=alt.Axis(labels=x_labels, title=x_title))
        x = alt.X(column, axis=alt.Axis(title=None), sort=sort_range)
    
    # to create chart
    chart = alt.Chart(chart_df, title=title).mark_bar(color='black')\
        .encode(x=x, 
                y=y)\
        .properties(height=300)
    
    # to choose whether or not to add text labels
    try: text = chart.mark_text(align='left', baseline='middle', dx=3).encode(text=chart_text)
    except: text= chart.mark_text(align='left', baseline='middle', dx=3)
    
    # create an expander
    expander = st.expander('',expanded=True)
    
    # to plot the chart
    expander.altair_chart(chart+text,use_container_width=True)
    expander.markdown(additional_text,unsafe_allow_html=True)

@st.cache_data(ttl=10*60)
def spree_line_chart(df, x='index', y='sse', title='SSE', color='#0033A0', point_size=50, show=True):
    
    # to create axes labels
    xaxis = alt.X(x, axis=alt.Axis(title=None))
    yaxis = alt.Y(y, axis=alt.Axis(title=title))
    
    # to plot chart
    line = alt.Chart(df).mark_line(color=color)\
            .encode(x=xaxis, y=yaxis, tooltip=[x,y])
    point = alt.Chart(df).mark_point(size=point_size,color=color)\
            .encode(x=xaxis, y=yaxis, tooltip=[x, y])
    chart = line+point
    
    # to show chart
    if show==True: st.altair_chart(chart, use_container_width=True)
    return chart

@st.cache_data(ttl=10*60)
def seg_line_chart(df, sort_order:list, x='Attribute', y='Avg Value', color='Cluster', height=400, show=True):
    # to plot chart
    chart = alt.Chart(df, title='Average Feature Value per Cluster').mark_line().encode(
        x=alt.X(x, axis=alt.Axis(title=None), sort=sort_order), y=y, color=color, tooltip=[x,y]).properties(height=height)
    # to show chart
    if show==True: st.altair_chart(chart, use_container_width=True)
    return chart

@st.cache_data(ttl=10*60)
def seg_heat_map(df, sort_order:list, x='Attribute', y='Cluster', height=400, show=True):
    # to remove respondent id if present
    df = df[df['Attribute'] != 'Respondent ID']
    
    # to plot and label chart
    chart = alt.Chart(df, title='Relative Feature Importance per Cluster').mark_rect().encode(x=alt.X('Attribute', axis=alt.Axis(title=None), sort=sort_order), y='Cluster', color='Value')
    text = chart.mark_text(baseline='middle').encode(
    text=alt.Text('Value:Q', format='.2f'),
    color=alt.condition(alt.datum.Value > 0, alt.value('white'), alt.value('black'))
)
    full_chart = (chart+text).properties(height=height)
    
    # to show chart
    if show == True: st.altair_chart(full_chart, use_container_width=True)
    return full_chart

def kmeans(df_features, min_range=1, max_range=11):
    # create standard scaler
    scaler = StandardScaler()
    # to fit scaler
    scaler.fit(df_features)
    # to normalise df
    df_norm = scaler.transform(df_features)
    df_norm = pd.DataFrame(df_norm, index=df_features.index, columns=df_features.columns)
    # to plot sses (sum of squared errors)
    sse= []
    for k in range(min_range,max_range):
        kmeans= KMeans(n_clusters=k,n_init= 10,max_iter=300,tol=0.0001)
        a= kmeans.fit(df_norm)
        sse.append(a.inertia_)
    sse = pd.DataFrame(sse, columns=['sse']).reset_index()
    return df_norm, sse

def kmeans_cust_seg(df_raw, df_norm, n_clusters):
    # to create clusters
    kmeans = KMeans(n_clusters, random_state=1)
    kmeans.fit(df_norm)
    cluster_labels = kmeans.labels_.astype(str)
    # to assign clusters to df
    df_norm = df_norm.assign(Cluster = cluster_labels)
    # to calculate average feature value per cluster
    df_melt = pd.melt(df_norm, id_vars='Cluster', var_name='Attribute', value_name='Avg Value')
    df_melt_chart = df_melt.groupby(['Attribute','Cluster']).mean().reset_index()
    # to calculate relative feature importance per cluster
    pop_avg = df_raw.mean()
    df_raw = df_raw.assign(Cluster = cluster_labels)
    cluster_avg = df_raw.groupby('Cluster').mean()
    relative_imp = cluster_avg/pop_avg - 1
    relative_imp_chart = pd.melt(relative_imp.reset_index(),id_vars='Cluster',var_name='Attribute',value_name='Value')
    relative_imp_chart['Cluster'] = relative_imp_chart['Cluster'].astype('category')
    return df_raw, df_norm, df_melt_chart, relative_imp_chart

def kmeans_segmentation(df_raw, cluster_result, sse, df_melt_chart, df_relative_imp_chart, seg_type='purchase behaviour'):
    # define sort order
    sort_order = list(df_raw.columns)
    # count number of customers per cluster and merge with cluster result
    cluster_number = df_raw['Cluster'].value_counts().to_frame().reset_index().rename(columns={'Cluster':'Number of customers','index':'Cluster'})
    cluster_result = cluster_result.merge(cluster_number,on='Cluster').iloc[:,[0,3,1,2]]
    # create sse expander
    spree_expander = st.expander(f'View {seg_type} KMeans SSE here')
    with spree_expander:
        spree_line_chart(sse) # show sse chart
    # create cluster result expander
    result_expander = st.expander('Feature Importance per Cluster', expanded=True)
    with result_expander:
        col1,col2 = st.columns(2)
        # create line chart of average feature value per cluster
        with col1: seg_line_chart(df_melt_chart, sort_order=sort_order)
        # create heat map of relative feature importance per cluster
        with col2: seg_heat_map(df_relative_imp_chart, sort_order=sort_order)
    st.write(cluster_result.set_index('Cluster'))
    return cluster_result

def kmodes(df_dict:dict, min_range=1, max_range=11):
    # plot sse for one to ten clusters for different dataframes in a dictionary
    df_sse = {}
    for df in df_dict.keys():
        sse=[]
        for k in range(min_range,max_range):
            kmodes= KModes(n_clusters=k,init='Huang', n_init=10)
            clusters= kmodes.fit(df_dict[df])
            sse.append(clusters.cost_)
        sse = pd.DataFrame(sse, columns=['sse']).reset_index()
        df_sse[df] = sse
    return df_sse

def kmodes_cust_seg(df_dict:dict, n_clusters):
    df_melt_chart_dict, df_relative_imp_chart_dict = ({} for _ in range(2))
    for df in df_dict.keys():
        pop_avg = df_dict[df].mean()
        # create kmodes clusters for specified number
        kmodes= KModes(n_clusters=n_clusters, init='Huang', n_init=10, random_state=1)
        clusters= kmodes.fit(df_dict[df])
        labels = kmodes.predict(df_dict[df])
        results = pd.DataFrame({'Cluster': labels}).astype(str)
        df_dict[df] = df_dict[df].assign(Cluster = results)
        # calculate average feature value per cluster
        df_melt = pd.melt(df_dict[df], id_vars='Cluster', var_name='Attribute', value_name='Avg Value')
        df_melt_chart_dict[df] = df_melt.groupby(['Attribute', 'Cluster']).mean().reset_index()
        cluster_avg = df_dict[df].groupby('Cluster').mean()
        # calculate relative feature importance per cluster
        relative_imp = cluster_avg/pop_avg - 1
        df_relative_imp_chart_dict[df] = pd.melt(relative_imp.reset_index(), id_vars='Cluster', var_name='Attribute', value_name='Value')
    return df_dict, df_melt_chart_dict, df_relative_imp_chart_dict

def list_customers(df, cols_to_select, select_cluster, seg_type='purchase_behaviour', key='key', text='', persona_type='FIS'):
    # create expander
    group_expander = st.expander(f'Expand this to view the personas')
    with group_expander:
        col1, col2 = st.columns([1,4])
        cluster_selection = col1.selectbox('Select cluster',options=select_cluster, key=key)
        # filter df to selected cluster
        if cluster_selection:
            df_cluster = df[df[f'{seg_type}_cluster']==cluster_selection][cols_to_select].set_index('Respondent ID')
            persona_df = df[[f'{seg_type}_cluster',f'{persona_type}_persona']]
            persona_df = persona_df[persona_df[f'{seg_type}_cluster']==cluster_selection]
            col1.markdown('<h4>Persona:</h4>',unsafe_allow_html=True)
            col1.write(persona_df.iloc[0,-1])
            col2.write(df_cluster)

def dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, persona, segments):
    # filter dfs to specified segments
    features_df = features_df[features_df[persona]==segments]
    channel_spend_sale_df = channel_spend_sale_df[channel_spend_sale_df[persona]==segments]
    benefits_df = benefits_df[benefits_df[persona]==segments]
    competitor_df = competitor_df[competitor_df[persona]==segments]
    return features_df, channel_spend_sale_df, benefits_df, competitor_df
