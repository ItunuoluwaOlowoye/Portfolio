import pandas as pd
import random
from PIL import Image
import streamlit as st
import os
from django.core.wsgi import get_wsgi_application
import udfs, ModelCalc # user defined functions in a py script of the same name

# set random seed for KMeans and KModes reproducibility
random.seed(1)

browser_tab_logo = Image.open('misc/pictures/browser-tab-logo.png') # store tab logo in a variable

# set page title, icon and layout
st.set_page_config(page_title='Customer Segmentation', page_icon=browser_tab_logo, layout='wide')

# bring all results from the ModelCalc calc_df script
customer_df, features_df, channel_spend_sale_df, benefits_df, competitor_df, id, gender_col, cat_age_col, num_age_col,\
cat_income_col, num_income_col, shop_for_col, high_freq_col, wwspend_col, saletype_col, benefits_col,comp_num_col,\
comp_spend_col, comp_list_col, used_app_col, prod_pref_list, channel_pref_list,bhv_df_raw, bhv_df_features,\
bhv_cluster_results, bhv_sse, bhv_df_melt_chart, bhv_df_relative_imp_chart, lty1_df_fns_raw, lty1_df_fns_features,\
lty1_cluster_result, lty1_df_fns_sse, lty1_df_fns_melt_chart,lty1_df_fns_relative_imp_chart, lty2_df_fs_raw,\
lty2_df_fs_features, lty2_cluster_result, lty2_df_fs_sse, lty2_df_fs_melt_chart, lty2_df_fs_relative_imp_chart,\
fns_customer_df, fs_customer_df, product_df_raw, product_df_features, product_cluster_result, product_df_sse,\
product_df_melt_chart, product_df_relative_imp_chart, products_df_sse, products_df_melt_chart_dict,\
products_df_relative_imp_chart_dict, products_df_list, channel_selected_cols, channel_clusters,\
channel_cluster_result = ModelCalc.calc_df()

# set up Django environment and application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings') 
application = get_wsgi_application()

placeholder = st.empty() # create a main content placeholder
with placeholder.container(): # create a container within the placeholder
    sb_placeholder = udfs.page_intro('Segmentation Model Output',"Please log in to access the model") # write the default page elements and store sidebar placeholder in a variable

# when authenticated
if udfs.authenticate_user(placeholder,sb_placeholder):

    # set logo image into a variable
    logo_image = 'misc/pictures/in-app-logo.png'
    
    # create sidebar with section options
    with st.sidebar:
        st.write('Select the section you want to see here.')
        report_type = st.selectbox('Sections', options=['Home Page', 'Dashboard', 'Customer Segments', 'Customer Profiles'])

    # for the home page
    if report_type=='Home Page':
        # set page header
        udfs.page_header('Customer Segmentation Model',logo_image)
        # give overview of the app
        st.markdown('''
        This segmentation model is built based on data provided from survey responses of 100 Walmart customers.
        These customers were asked questions about their biodata, and their interactions with Walmart and its
        competitors over three months. The responses were recorded in February and March, 2023, although one
        customer responded in November 2022.

        In this app, there are three sections. You can select the section to view in the sidebar to the left.
        1. A dashboard with insights on the customer base profiles and their interactions with the store
        2. Customer segments and their characteristics
        3. Customer personas built from the different customer segments identified
        
        There are four categories used to build customer segments:
        1. **Behavioural Segmentation**: This is based on their purchasing
        behaviour. It takes into account how frequently they shop with Walmart, how much they earn, and 
        how much they spend. This segmentation can identify customers to be targeted with __*specific marketing 
        campaigns*__ tailored towards their needs
        2. **Competitor Landscape/Loyalty Segmentation**: This is based on whether or not customers prefer to shop 
        at Walmart compared to shopping at competitor stores. In understanding the competitor landscape, we can
        infer how loyal customers are to the Walmart brand. It takes into account how frequently they shop with
        Walmart, how many listed competitor stores they also shop from, and how much they spend at these stores.
        This can help to identify potential areas for improvement, and develop targeted strategies to __*enhance
        customer satisfaction and loyalty*__ where needed.
        3. **Product Preference Segmentation**: This is based on preference for the four product offerings (food, 
        fashion, home, beauty and health). It takes into account how frequently they shop for the different products.
        Each group identified from this segmentation can be targeted with __*personalized recommendations*__ tailored 
        towards their needs.
        4. **Channel Preference Segmentation**: This is based on customers preferred mode of shopping, whether instore,
        through the app, or online. Preferences vary for the different product offerings. For example, a customer can
        prefer to buy food instore but buy fashion items online. Therefore, this segmentation is broken down by the
        channel preference for each product. This helps in creating optimization strategies for __*improving customer
        experience*__.
        ''', unsafe_allow_html=True,)

    elif report_type=='Dashboard':
        # set page header
        udfs.page_header('Customer Base Insights',logo_image)
        
        # create filter by segment groups and personas
        st.markdown('''Here, the features of the customer base are visualized. By default, the dashboard shows the
            features for all customers. In the drop-downs and buttons below, you can choose to filter to customers 
            within particular segments.''', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap='small')
        groups = col1.selectbox('What customer group would you like to see?',options=['All customers','Purchase behaviour','Competitor landscape/Loyalty','Product Preference','Channel Preference'])
        
        # create filters for the different segment groups
        if groups == 'Purchase behaviour':
            segments = col2.radio('Select customer segment', options=customer_df['FIS_persona'].unique(), horizontal=True)
            features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'FIS_persona', segments)
        elif groups == 'Competitor landscape/Loyalty':
            segments = col2.radio('Select customer segment', options=customer_df['FNs_persona'].unique(), horizontal=True)
            features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'FNs_persona', segments)
        elif groups == 'Product Preference':
            segments = col2.radio('Select customer segment', options=customer_df['PF_persona'].unique(), horizontal=True)
            features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'PF_persona', segments)
        elif groups == 'Channel Preference':
            products = col2.selectbox('Which product channel would you like to see?', options=['Food','Fashion','Home','Beauty and Health'])
            if products == 'Food':
                segments = col2.radio('Select customer segment', options=customer_df['food_channel_persona'].unique(), horizontal=True)
                features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'food_channel_persona', segments)
            elif products == 'Fashion':
                segments = col2.radio('Select customer segment', options=customer_df['fashion_channel_persona'].unique(), horizontal=True)
                features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'fashion_channel_persona', segments)
            elif products == 'Home':
                segments = col2.radio('Select customer segment', options=customer_df['home_channel_persona'].unique(), horizontal=True)
                features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'home_channel_persona', segments)
            else:
                segments = col2.radio('Select customer segment', options=customer_df['beauty_channel_persona'].unique(), horizontal=True)
                features_df, channel_spend_sale_df, benefits_df, competitor_df =\
                udfs.dashboard_filter_group(features_df, channel_spend_sale_df, benefits_df, competitor_df, 'beauty_channel_persona', segments)
        
        # create tabs for the different insight categories and generate the charts
        demography, behaviour, loyalty, product_preference, channel_preference = st.tabs(['Demography', 'Purchase Behaviour', 'Competitor Landscape/Customer Loyalty', 'Product Preference', 'Channel Preference'])
        with demography:
            st.markdown(f'''This tab summarizes the gender, age, and income distribution of customers''', 
                        unsafe_allow_html=True)
            individual_features, features_rltnship = st.tabs(['Individual Features','Relationship between Features'])
            with individual_features:
                col1, col2, col3 = st.columns(3, gap='small')
                with col1:
                    udfs.dashboard_bar_charts(features_df[[gender_col]], gender_col, gender_col, hide_record='prefer not to say')
                with col2:
                    udfs.dashboard_bar_charts(features_df[[cat_income_col]], cat_income_col, 'Annual household income')
                with col3:
                    udfs.dashboard_bar_charts(features_df[[cat_age_col]], cat_age_col, 'Age')
            with features_rltnship:
                col1, col2 = st.columns(2, gap='small')
                with col1:
                    udfs.dashboard_bar_charts(features_df[[gender_col,num_income_col]], gender_col, 'Median income per gender', type='bivariate', x_labels=True, x_title=num_income_col, hide_record='prefer not to say')
                with col2:
                    udfs.dashboard_bar_charts(features_df[[cat_age_col,num_income_col]], cat_age_col, 'Median income per age group', type='bivariate', x_labels=True, x_title=num_income_col)
        with behaviour:
            st.markdown('''This tab summarizes the purchase behaviour of customers such as:<br>
                        1. Who they shop for.<br>
                        2. Whether they prefer to buy fullprice or discounted items. **Please note that each customer can choose more
                        than one option**.<br>
                        3. How often they shop: In the dataset, the shopping frequency was provided for each product offering. The maximum 
                        frequency was selected as the frequency of shopping from Walmart in general.<br>
                        4. Whether or not they use store benefits. **Please note that each customer can use more than one store
                        benefit**.<br>
                        5. How much they spend per shop.''', 
                        unsafe_allow_html=True)
            individual_features, features_rltnship = st.tabs(['Individual Features','Relationship between Features'])
            with individual_features:
                col1, col2, col3 = st.columns(3, gap='small')
                with col1:
                    udfs.dashboard_bar_charts(features_df[[shop_for_col]], shop_for_col, 'Who do customers shop for?')
                    udfs.dashboard_bar_charts(channel_spend_sale_df[[id, saletype_col]], saletype_col, 'Do they buy fullprice or onsale?', dformat='unpivoted')
                with col2:
                    udfs.dashboard_bar_charts(features_df[[high_freq_col]], high_freq_col, 'How often do they shop?')
                    udfs.dashboard_bar_charts(benefits_df[[id,benefits_col,num_age_col]], benefits_col, 'What store benefits do they use?', dformat='unpivoted',hide_record="I’m not sure")
                with col3:
                    udfs.dashboard_bar_charts(channel_spend_sale_df[[id, wwspend_col]], wwspend_col, 'How much do they spend per shop?', dformat='unpivoted')
            with features_rltnship:
                col1, col2 = st.columns(2, gap='small')
                with col1:
                    udfs.dashboard_bar_charts(features_df[[num_age_col, shop_for_col]], shop_for_col, 'Median age for various shopping reasons', type='bivariate', x_labels=True, x_title=num_age_col)
                with col2:
                    udfs.dashboard_bar_charts(benefits_df[[id,benefits_col,num_age_col]], benefits_col, 'Median age for benefits usage', type='bivariate', x_labels=True, x_title=num_age_col, dformat='unpivoted',hide_record="I’m not sure")
        with loyalty:
            st.markdown(f'''This tab summarizes the competitor landscape and can be used to infer customer loyalty
                        through how many competitors each customer shops from and a comparison of their spend at
                        Walmart to their spend at competitors. <br>**Please note that the list of competitors is
                        not exhaustive. Customers responded `None of the above` if the competitor was not listed.
                        `None of the above` responses were counted as one possible competitor**''', 
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap='small')
            with col1:
                udfs.dashboard_bar_charts(competitor_df[[id,comp_list_col]], comp_list_col, 'Top rated competitors', dformat='unpivoted', limit=True, limit_number=7)
                udfs.dashboard_bar_charts(features_df[[comp_spend_col]], comp_spend_col, 'Customer spend at competitors', hide_record='No value')
            with col2:
                udfs.dashboard_bar_charts(features_df[[comp_num_col]].astype(str), comp_num_col, 'Number of listed competitors customers shop with', limit=True, limit_number=7)
        with product_preference:
            st.markdown(f'''This tab summarizes how often customers shop for the different product offerings - Food,
                        Fashion, Home, and Beauty&Health''', 
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                udfs.dashboard_bar_charts(features_df[[prod_pref_list[0]]], prod_pref_list[0], 'How often do customers shop for '+prod_pref_list[0].split(':')[1]+'?')
                udfs.dashboard_bar_charts(features_df[[prod_pref_list[2]]], prod_pref_list[2], 'How often do customers shop for '+prod_pref_list[2].split(':')[1]+'?')
            with col2:
                udfs.dashboard_bar_charts(features_df[[prod_pref_list[1]]], prod_pref_list[1], 'How often do customers shop for '+prod_pref_list[1].split(':')[1]+'?')
                udfs.dashboard_bar_charts(features_df[[prod_pref_list[3]]], prod_pref_list[3], 'How often do customers shop for '+prod_pref_list[3].split(':')[1]+'?')
        with channel_preference:
            st.markdown(f'''This tab summarizes the channel customers prefer to shop for the different product
                        offerings. **Please note that customers can prefer more than one channel**''', 
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                udfs.dashboard_bar_charts(channel_spend_sale_df[[id, channel_pref_list[0]]], channel_pref_list[0], 'How do customers most often shop for '+channel_pref_list[0].split(':')[1]+'?', dformat='unpivoted', hide_record='Not Applicable')
                udfs.dashboard_bar_charts(channel_spend_sale_df[[id, channel_pref_list[2]]], channel_pref_list[2], 'How do customers most often shop for '+channel_pref_list[2].split(':')[1]+'?', dformat='unpivoted', hide_record='Not Applicable')
                udfs.dashboard_bar_charts(features_df[[used_app_col]], used_app_col, used_app_col)
            with col2:
                udfs.dashboard_bar_charts(channel_spend_sale_df[[id, channel_pref_list[1]]], channel_pref_list[1], 'How do customers most often shop for '+channel_pref_list[1].split(':')[1]+'?', dformat='unpivoted', hide_record='Not Applicable')
                udfs.dashboard_bar_charts(channel_spend_sale_df[[id, channel_pref_list[3]]], channel_pref_list[3], 'How do customers most often shop for '+channel_pref_list[3].split(':')[1]+'?', dformat='unpivoted', hide_record='Not Applicable')

    elif report_type=='Customer Segments':
        # set the page header
        udfs.page_header('Customer Segments',logo_image)
        # create tabs for the different segment groups
        seg_behaviour, seg_loyalty, seg_product_preference, seg_channel_preference = st.tabs(['Behavioural Segmentation', 'Loyalty Segmentation', 'Product Preference Segmentation', 'Channel Preference segmentation'])
        
        # create overview, cluster xter, insights and recommendations for each segment groups
        with seg_behaviour:
            overview, cluster, rec = st.tabs(['Overview', 'Clustering Algorithm', 'Insights and Recommendations'])
            with overview:
                st.markdown(f'''**Behavioural Segmentation**: This is based on customer purchasing behaviour. This segmentation can 
                            identify customers to be targeted with __*specific marketing campaigns*__ tailored
                            towards their needs.<br>The KMeans algorithm was used to create clusters/personas based on: <br> 
                            1. How often customers shop at Walmart *(general_freq)* <br> 2. How much they earn
                            *(income)*<br> 3. How much they spend *(Spend at Walmart)*
                            when shopping at Walmart.<br><br>These features were preprocessed for the algorithm in the following manner:<br>
                            1. Frequency, income, and spend at Walmart were all converted to numeric values in
                            one column each. For income and spend, the average of the ranges given was used as the 
                            numeric value; for frequency, the timeframes were converted to numbers<br>
                            2. The distributions of frequency and income were validated to be symmetric and the spend at Walmart was 
                            converted to logarithmic scale so that the distribution can become symmetric<br>
                            3. All features were standardized to an average of 0 and variance of 1<br>
                            4. The algorithm was run to create from 1 to 10 clusters, calculating sum of squared errors (SSE) and selecting
                            the number of clusters with relatively small SSEs as the best model<br>
                            5. The **average value of each feature for each cluster was plotted on a line graph** and the **relative importance
                            of these features in each cluster was plotted on a heat map** to identify differences between clusters.<br>
                            6. These differences were noted and used to create personas for the customers.''',
                            unsafe_allow_html=True)
            with cluster:
                bhv_cluster_results = udfs.kmeans_segmentation(bhv_df_raw, bhv_cluster_results, bhv_sse, bhv_df_melt_chart, bhv_df_relative_imp_chart)
            with rec:
                st.write(bhv_cluster_results.set_index('Cluster'))
                cols_to_select = [id,high_freq_col,cat_income_col] + [col for col in customer_df.columns if 'How much do you normally spend, per shop, when you shop at Walmart' in col]
                udfs.list_customers(customer_df, cols_to_select, bhv_cluster_results['Cluster'].unique())
                st.markdown('<h4> Insights and Recommendations </h4>', unsafe_allow_html=True)
                st.markdown('''
                **Cluster 0**: These set of customers have the potential to either spend more or buy more often.
                They should be targeted for marketing campaigns<br>
                **Cluster 1**: These are challenging customers who are likely to churn<br>
                **Cluster 2**: These customers have the potential to buy more often although may be inhibited by income. They
                can be targeted for promotional offers and discounts to encourage them to buy more<br>
                **Cluster 3**: These are the customers with the best purchasing behaviours. They buy very often, spend
                reasonably and also earn good income. They can also be targeted for marketing campaigns''',
                unsafe_allow_html=True)
                
        with seg_loyalty:
            overview, cluster, rec = st.tabs(['Overview', 'Clustering Algorithm', 'Customer Personas, Insights and Recommendations'])
            with overview:
                st.markdown(f'''**Competitor Landscape/Loyalty Segmentation**: This is based on whether or not customers 
                            prefer to shop at Walmart compared to shopping at competitor stores. In understanding
                            the competitor landscape, we can infer how loyal customers are to the Walmart brand.
                            This can help to identify potential areas for improvement, and develop targeted strategies
                            to __*enhance customer satisfaction and loyalty*__ where needed.<br><br>The KMeans algorithm
                            was used to create clusters/personas based on: <br> 
                            1. How often customers shop at Walmart *(general_freq)* <br> 2. The number of listed competitors
                            they also shop from *(Number of competitors)*. Please note that for customers who answered `None of
                            the above`, it was taken to mean at least one other possible competitor. <br> 3. The ratio of how 
                            much they spend at Walmart compared to how much they spend at competitors *(spend_ratio_ww_vs_comp)*.
                            It was calculated as $$ Spend at Walmart/Spend at competitors $$. A low spend ratio means
                            that they spent more at competitors while a high spend ration means that they spent more at 
                            Walmart. <br><br>For customers who did not respond to how much they spent at competitors, their frequency and number of
                            competitors were used to infer their loyalty to Walmart.<br><br>These features were 
                            preprocessed for the algorithm in the following manner:<br>
                            1. The spend ratio and number of competitors was calculated; for frequency, the timeframes
                            were converted to numbers<br>
                            2. The distribution of frequency was validated to be symmetric. For the others to be symmetric,
                            the number of competitors was converted to a cubic distribution while the spend ratio was converted
                            to a logarithmic distribution<br>
                            3. All features were standardized to an average of 0 and variance of 1<br>
                            4. The algorithm was run to create from 1 to 10 clusters, calculating sum of squared errors (SSE) and selecting
                            the number of clusters with relatively small SSEs as the best model<br>
                            5. The **average value of each feature for each cluster was plotted on a line graph** and the **relative importance
                            of these features in each cluster was plotted on a heat map** to identify differences between clusters.<br>
                            6. These differences were noted and used to create personas for the customers.''',
                            unsafe_allow_html=True)
            with cluster:
                first_group, second_group = st.tabs(['Customers who disclosed how much they spent at competitors', 'Customers who did not disclose how much they spent at competitors'])
                with first_group:
                    lty1_cluster_result = udfs.kmeans_segmentation(lty1_df_fns_raw, lty1_cluster_result, lty1_df_fns_sse, lty1_df_fns_melt_chart, lty1_df_fns_relative_imp_chart, seg_type='loyalty')
                with second_group:
                    lty2_cluster_result = udfs.kmeans_segmentation(lty2_df_fs_raw, lty2_cluster_result, lty2_df_fs_sse, lty2_df_fs_melt_chart, lty2_df_fs_relative_imp_chart, seg_type='loyalty')
            with rec:
                first_group, second_group = st.tabs(['Customers who disclosed how much they spent at competitors', 'Customers who did not disclose how much they spent at competitors'])
                with first_group:
                    persona_df = lty1_cluster_result.set_index('Cluster')
                    persona_df.columns = persona_df.columns.str.replace('FNS_','').str.title()
                    st.write(persona_df)
                    cols_to_select = ['customer_id','Thinking back over the last 3 months, how regularly did you shop with Walmart', 'Number of competitors']
                    udfs.list_customers(fns_customer_df, cols_to_select+ ['spend_ratio_ww_vs_comp'], lty1_cluster_result['Cluster'].unique(), seg_type='loyalty1',key='loyalty1', persona_type='FNS')
                    st.markdown('<h4> Insights and Recommendations </h4>', unsafe_allow_html=True)
                    st.markdown('''
                    **Cluster 0**: These set of customers have alternatives to Walmart and spend more at the competitors.
                    They may be difficult to convert.<br>
                    **Cluster 1**: These customers have other alternatives but still shop often at Walmart. Strategies can
                    be implemented based on dashboard insights from these customers to enhance customer satisfaction and stock up more
                    of what they prefer to shop<br>
                    **Cluster 2**: These customers have many alternatives and will be difficult to convert<br>
                    **Cluster 3**: These customers spend more at Walmart and have lesser alternatives. They don't shop as often and can be
                    targeted for upselling opportunities<br>
                    **Cluster 4**: These are loyal customers and can be targeted for exclusive loyalty programs''',
                    unsafe_allow_html=True)
                with second_group:
                    persona_df = lty2_cluster_result.set_index('Cluster')
                    persona_df.columns = persona_df.columns.str.replace('FN_','').str.title()
                    st.write(persona_df)
                    udfs.list_customers(fs_customer_df, cols_to_select, lty2_cluster_result['Cluster'].unique(), seg_type='loyalty2', key='loyalty2', persona_type='FN')
                    st.markdown('<h4> Insights and Recommendations </h4>', unsafe_allow_html=True)
                    st.markdown('''
                    **Cluster 0**: These set of customers shop often from Walmart although they may have few to many
                    alternatives. They can also be eligible for loyalty programs to keep them devoted to Walmart.<br>
                    **Cluster 1**: These customers show no visible preference for either Walmart or its competitors<br>
                    **Cluster 2**: These customers have many alternatives and will be difficult to convert''',
                    unsafe_allow_html=True)
        
        with seg_product_preference:
            overview, cluster, rec = st.tabs(['Overview', 'Clustering Algorithm', 'Customer Personas, Insights and Recommendations'])
            with overview:
                st.markdown(f'''**Product Preference Segmentation**: This is based on customer preference for the 
                            four product offerings (food, fashion, home, beauty and health). It takes into account 
                            how frequently they shop for the different products. Each group identified from this 
                            segmentation can be targeted with __*personalized recommendations*__ tailored towards
                            their needs.<br>The KMeans algorithm was used to create clusters/personas based on how
                            often they buy any of food, fashion, home, or beauty and health items.<br><br>
                            These features were preprocessed for the algorithm in the following manner:<br>
                            1. The frequencies for each product offering were all converted to numeric values in
                            one column each.<br>
                            2. The frequency distributions were validated to be symmetric<br>
                            3. All features were standardized to an average of 0 and variance of 1<br>
                            4. The algorithm was run to create from 1 to 10 clusters, calculating sum of squared errors (SSE) and selecting
                            the number of clusters with relatively small SSEs as the best model<br>
                            5. The **average value of each feature for each cluster was plotted on a line graph** and the **relative importance
                            of these features in each cluster was plotted on a heat map** to identify differences between clusters.<br>
                            6. These differences were noted and used to create personas for the customers.''',
                            unsafe_allow_html=True)
            with cluster:
                product_cluster_result = udfs.kmeans_segmentation(product_df_raw, product_cluster_result, product_df_sse, product_df_melt_chart, product_df_relative_imp_chart, seg_type='product preference')
            with rec:
                st.write(product_cluster_result.set_index('Cluster'))
                cols_to_select = [id] + [col for col in customer_df.columns if 'Thinking back over the last 3 months, how regularly did you shop with Walmart for different products:' in col]
                udfs.list_customers(customer_df, cols_to_select, product_cluster_result['Cluster'].unique(), key='prod_pref', seg_type='product_preference', persona_type='PF')
                st.markdown('<h4> Insights and Recommendations </h4>', unsafe_allow_html=True)
                st.markdown('''
                **Cluster 0**: These customers should be targeted for food promotions<br>
                **Cluster 1**: These customers should be targeted for food promotions and upselling opportunities<br>
                **Cluster 2**: These customers should be targeted for fashion promotions and upselling opportunities in food,
                and beauty and helth products.<br>
                **Cluster 3**: These are challenging customers rarely buy any product offerings<br>
                **Cluster 4**: These customers should be targeted for home promotions and upselling opportunities in fashion''',
                unsafe_allow_html=True)
                
        with seg_channel_preference:
            overview, cluster, persona, rec = st.tabs(['Overview', 'Clustering Algorithm', 'Customer Personas', 'Insights and Recommendations'])
            with overview:
                st.markdown(f'''**Channel Preference Segmentation**: This is based on customers preferred mode of 
                            shopping, whether instore, through the app, or online. Preferences vary for the different
                            product offerings. This segmentation is broken down by the channel preference for each 
                            product. This helps in creating optimization strategies for __*improving customer experience*__.<br>
                            The KModes algorithm was used to create clusters/personas based on the channel preference for the different
                            product offerings. <br><br>These features were preprocessed for the algorithm in the following manner:<br>
                            1. The responses for the different channels were recording in different columns and these columns were one-hot
                            encoded into binary values (either 0(false) or 1(true)).<br>
                            2. The algorithm was run to create from 1 to 10 clusters, calculating sum of squared errors (SSE) and selecting
                            the number of clusters with relatively small SSEs as the best model<br>
                            5. The **average value of each feature for each cluster was plotted on a line graph** and the **relative importance
                            of these features in each cluster was plotted on a heat map** to identify differences between clusters.<br>
                            6. These differences were noted and used to create personas for the customers.''',
                            unsafe_allow_html=True)
            with cluster:
                for products_df in products_df_list.keys():
                    st.markdown(f'##### {products_df}')
                    with st.expander(f'View {products_df} channel KModes SSE here'):
                        chart = udfs.spree_line_chart(products_df_sse[products_df], show=True)
                    with st.expander(f'Feature Importance per Cluster', expanded=True):
                        col1,col2 = st.columns(2)
                        with col1:
                            linechart_seg = udfs.seg_line_chart(products_df_melt_chart_dict[products_df], sort_order=list(products_df_list[products_df].columns))
                        with col2:
                            heatmap_seg = udfs.seg_heat_map(products_df_relative_imp_chart_dict[products_df], sort_order=list(products_df_list[products_df].columns))
                    cluster_number = products_df_list[products_df]['Cluster'].value_counts().to_frame().reset_index().rename(columns={'Cluster':'Number of customers','index':'Cluster'})
                    channel_cluster_result[products_df] = channel_cluster_result[products_df].merge(cluster_number,on='Cluster')
                    st.write(channel_cluster_result[products_df].iloc[:,[0,3,1,2]].set_index('Cluster'))
                    st.divider()
            with persona:
                for products_df in products_df_list:
                    st.markdown(f'##### {products_df}')
                    cols_to_select = [id] + [col for col in channel_selected_cols if products_df in col]
                    udfs.list_customers(customer_df, cols_to_select, channel_clusters, seg_type=products_df.lower()+'_channel', key=products_df, text=products_df, persona_type=products_df.lower()+'_channel')
                    st.divider()
            with rec:
                food, fashion, home, beauty = st.tabs(['Food','Fashion','Home','Beauty and Health'])
                with food:
                    st.write(channel_cluster_result['Food'].iloc[:,[0,3,1,2]].set_index('Cluster'))
                    st.markdown('''
                    **Cluster 0**: These customers like both personalized service and convenience. Promotions can be
                    offered to them both instore and online<br>
                    **Cluster 1**: These customers prefer personalized service although they can occasionally use the
                    app. Instore promotions should be offered to them. Also, if Walmart wants to drive more traffic towards the 
                    app for food sales, these customers can be targeted<br>
                    **Cluster 2**: These customers prefer convenience and use the app. App-specific promotions can be offered
                    to them. Since they sometimes shop instore, they can also be eligible for instore promotions<br>
                    **Cluster 3**: These customers prefer personalised service and can be targeted for in-store promotions''',
                    unsafe_allow_html=True)
                with fashion:
                    st.write(channel_cluster_result['Fashion'].iloc[:,[0,3,1,2]].set_index('Cluster'))
                    st.markdown('''
                    **Cluster 0**: These customers like both personalized service and convenience. Promotions can be
                    offered to them both instore and online<br>
                    **Cluster 1**: These customers prefer online service and can be targeted for online promotions<br>
                    **Cluster 2**: These customers rarely use any of the channels<br>
                    **Cluster 3**: These customers prefer personalised service and can be targeted for in-store promotions''',
                    unsafe_allow_html=True)
                with home:
                    st.write(channel_cluster_result['Home'].iloc[:,[0,3,1,2]].set_index('Cluster'))
                    st.markdown('''
                    **Cluster 0**: These customers rarely use any of the channels<br>
                    **Cluster 1**: These customers prefer personalized service although they can occasionally use the
                    app. Instore promotions should be offered to them. Also, if Walmart wants to drive more traffic towards the 
                    app for fashion sales, these customers can be targeted<br>
                    **Cluster 2**: These customers prefer personalised service and can be targeted for in-store promotions<br>
                    **Cluster 3**: These customers prefer convenience and shopping online so they can be targeted for online promotions''',
                    unsafe_allow_html=True)
                with beauty:
                    st.write(channel_cluster_result['Beauty'].iloc[:,[0,3,1,2]].set_index('Cluster'))
                    st.markdown('''
                    **Cluster 0**: These customers prefer personalised service and can be targeted for in-store promotions<br>
                    **Cluster 1**: These customers prefer convenience and use the app most often. They can be targeted for app-specific
                    promotions<br>
                    **Cluster 2**: These customers prefer convenience and shop online most often. They can be targeted for online
                    promotions<br>
                    **Cluster 3**: These customers rarely uses any of the channels''',
                    unsafe_allow_html=True)

    # for the personas
    else:
        # set header
        udfs.page_header('Customer Profiles',logo_image)
        
        #......................................................................................
        # create df for profile and store feature values in variables
        seg_df = customer_df.astype(str).replace('nan','')
        seg_df['customer_id'] = seg_df['customer_id'].astype(str)
        seg_columns = list(seg_df.columns)
        name_col, filter_col = st.columns([1,2])
        name_col.write('Whose profile do you want to check?')
        customer_id_list = list(customer_df['customer_id'].unique())
        index_person = '118251975590'
        try: 
            customer_id_list.remove(index_person)
            customer_id_list = [index_person] + customer_id_list
        except: pass
        customer = filter_col.selectbox('Customer ID', options=customer_id_list, label_visibility='collapsed')
        seg_df = seg_df[seg_df['customer_id']==customer].fillna('')
        gender = seg_df.iloc[0]['gender']
        age = seg_df.iloc[0]['age_range']
        shop_for = seg_df.iloc[0]['Who do you normally shop for?']
        income_orig = seg_df.iloc[0]['What is your total annual household income? Untreated']
        income_filled = seg_df.iloc[0]['What is your total annual household income?']
        if income_orig == '': income_orig = income_filled + '(predicted)'
        saletype = seg_df[['How would you describe the fashion/ home/beauty items you usually buy?All full price items',
                        'How would you describe the fashion/ home/beauty items you usually buy?Mostly full price, with some sale / discount items',
                        'How would you describe the fashion/ home/beauty items you usually buy?Mostly sale / discount items, with some full price items',
                        'How would you describe the fashion/ home/beauty items you usually buy?All sale / discount items']]#.bfill(axis=1).iloc[:,0]
        saletype = saletype.fillna('').apply(lambda row: ', '.join(row.astype(str)), axis=1).str.strip(', ').str.replace(', , , ',', ').str.replace(', , ',', ')
        benefits = seg_df.loc[:,[col for col in seg_columns if 'Do you have any of the following? Please tick all the apply' in col]]
        benefits = benefits.fillna('').apply(lambda row: ', '.join(row.astype(str)), axis=1).str.strip(', ').str.replace(', , , ',', ').str.replace(', , ',', ').str.replace(', , ',', ')
        competitors = seg_df.loc[:,[col for col in seg_columns if 'Where else do you shop for food and clothing online, if not Walmart?' in col and 'None of the above' not in col]].fillna('')
        competitors = pd.melt(competitors)['value']
        competitors = competitors.unique().tolist()
        spend_competitor = seg_df.iloc[0]['How much do you spend at these places each month?Open-Ended Response']
        if spend_competitor == 'No value': spend_competitor = 'not recorded'
        freq = seg_df.iloc[0]['Thinking back over the last 3 months, how regularly did you shop with Walmart']
        app_shop = seg_df.iloc[0]['Have you used the Walmart app to shop?']
        num_competitor = seg_df.iloc[0]['Number of competitors']
        if num_competitor==1: competitor_text='competitor'
        else: competitor_text='competitors'
        ww_spend_orig = seg_df.iloc[0]['Spend at Walmart Untreated Range']
        ww_spend_filled = seg_df.iloc[0]['Spend at Walmart Range']
        if ww_spend_orig == '': ww_spend_orig=ww_spend_filled + '(predicted)'
        purchase_behaviour = seg_df.iloc[0]['FIS_persona']
        loyalty = seg_df.iloc[0]['FNs_persona']
        product_preference = seg_df.iloc[0]['PF_persona']
        food_channel_preference = seg_df.iloc[0]['food_channel_persona']
        fashion_channel_preference = seg_df.iloc[0]['fashion_channel_persona']
        home_channel_preference = seg_df.iloc[0]['home_channel_persona']
        beauty_channel_preference = seg_df.iloc[0]['beauty_channel_persona']
        #................................................................................................

        # create header with customer id
        st.markdown(f"""
        <div style='background-color: black; padding: 0px; color: white; font-size: 36px; font-weight: bold;'>{customer}</div>
        """,unsafe_allow_html=True)
        st.write('')
        
        # set three columns
        col1, col2, col3 = st.columns(3, gap='small')
        
        #COLUMN 1.......................................
        # add the age, income, spend at Walmart and benefits to column 1
        with col1:
            st.markdown('''<h6 style="color:#0033A0;">Bio</h6>''', unsafe_allow_html=True)
            st.markdown(
        f"""
        <div style='display: flex; justify-content: space-between; font-size: 14px; padding: 2px;'>
            <div style='text-align: left; font-weight: bold;'>Age: </div>
            <div style='text-align: right;'>{age}</div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 14px;'>
            <div style='text-align: left; font-weight: bold;'>Income: </div>
            <div style='text-align: right;'>{income_orig}</div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 14px;'>
            <div style='text-align: left; font-weight: bold;'>Spend: </div>
            <div style='text-align: right;'>{ww_spend_orig}</div>   
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 14px;'>
            <div style='text-align: left; font-weight: bold;'>Benefits: </div>
            <div style='text-align: right;'>{benefits.iloc[0]}</div> 
        </div>
        """,
        unsafe_allow_html=True
    )
            # choose image according to gender
            if gender == 'Female': 
                image = Image.open('misc/pictures/female.jpg'); pronoun = 'She'
            else: image = Image.open('misc/pictures/male.jpg'); pronoun= 'He'
            st.write('')
            # write the image
            st.image(image)
            # write the shopping style of the customer
            st.markdown(f'''<h6 style="color:#0033A0;">Shopping style</h6>
                    <font style="font-size:14px;">
                        {customer} shops from Walmart <font><strong>{freq.lower()}</strong></font> in three months. <br>
                        {pronoun} says and I quote, "I usually shop for <font><strong>{shop_for}</strong></font>."
                        </font><br><br>''', unsafe_allow_html=True)
        
        # COLUMN 2.......................................
        with col2:
            # write the purchase behaviour, app usage, and channel preference of the customer in column 2
            # also include competitive landscape/loyalty
            st.markdown('''<h6 style="color:#0033A0;">Purchase behaviour</h6>''', unsafe_allow_html=True)
            st.markdown(f'''
                        <font style="font-size:14px;">
                        {pronoun} is a {purchase_behaviour}
                        </font><br><br>
                        <h6 style="color:#0033A0;">Preferred shopping channels</h6>
                        <font style="font-size:14px;">
                        Has {pronoun.lower()} used the app to shop before? <strong>{app_shop}</strong><br>
                        <strong>For food:</strong> {pronoun} {food_channel_preference}.<br>
                        <strong>For beauty:</strong> {pronoun} {beauty_channel_preference}.<br>
                        <strong>For fashion:</strong> {pronoun} {fashion_channel_preference}.<br>
                        <strong>For home:</strong> {pronoun} {home_channel_preference}.
                        </font><br><br>
                        <h6 style="color:#0033A0;">Competitive play</h6>
                        <font style="font-size:14px;">
                        {pronoun} {loyalty}. {pronoun} shops at <strong>{num_competitor} listed {competitor_text}</strong> 
                        and the amount of money {pronoun.lower()} spends at these stores is <strong>{spend_competitor}.</strong>
                        </font><br><br>
                        ''', unsafe_allow_html=True)
        
        # COLUMN 3.......................................
        with col3:
            # write the products preferred by the customers
            st.markdown('''<h6 style="color:#0033A0;">Preferred products</h6>''', unsafe_allow_html=True)
            st.markdown(f'''<font style="font-size:14px;">
                        {pronoun} is <font>{product_preference}</font>
                        </font><br><br>''', unsafe_allow_html=True)
            # also create an expander with the logos of competitors the customer frequents
            with st.expander(f'Other brands {pronoun.lower()} trusts include:', expanded=True):
                st.write()
                images_list = []
                for picture in competitors:
                    if picture != '':
                        try: image = Image.open(f'misc/pictures/competitors/{picture}.png')
                        except: 
                            try: image = Image.open(f'misc/pictures/competitors/{picture}.jpg')
                            except: st.markdown(f'*{picture}* logo could not be loaded')
                        images_list.append(image)
                num_columns = 2
                num_images_per_column = len(images_list) // num_columns
                columns = st.columns(num_columns)
                for i, column in enumerate(columns):
                    start_index = i * num_images_per_column
                    end_index = (i+1) * num_images_per_column
                    for image in images_list[start_index:end_index]:
                        column.image(image)
        st.divider()