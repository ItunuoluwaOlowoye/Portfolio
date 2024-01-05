import numpy as np
import pandas as pd
import random
import numpy as np
from PIL import Image
from kmodes.kmodes import KModes
from oauth2client.service_account import ServiceAccountCredentials
import gspread as gs
import streamlit as st
import udfs # user defined functions in a py script of the same name

# set random seed for KMeans and KModes reproducibility
random.seed(1)

# for connecting to google sheets
gs_credentials = ServiceAccountCredentials.from_json_keyfile_name('misc/serviceacct.json')

@st.cache_data(ttl=10*60)
def calc_df():
    # clean up the data
    customer_df = udfs.data_transform_phase_one('phase2.csv')
    # create dataframes for dashboard
    features_df, channel_spend_sale_df, benefits_df, competitor_df = udfs.data_transform_phase_two(customer_df)
    # put column names into variables
    id, gender_col, cat_age_col, num_age_col, cat_income_col, num_income_col, shop_for_col, high_freq_col, \
        wwspend_col, saletype_col, benefits_col, comp_num_col, comp_spend_col, comp_list_col, used_app_col, prod_pref_list, channel_pref_list\
            = 'Respondent ID', 'Gender', 'How old are you?', 'age', 'What is your total annual household income?', 'income', 'Who do you normally shop for?', 'Thinking back over the last 3 months, how regularly did you shop with Woolworths',\
                'How much do you normally spend, per shop, when you shop at Woolworths?', 'How would you describe the fashion/ home/beauty items you usually buy?',\
                    'Do you have any of the following? Please tick all the apply', 'Number of competitors', 'How much do you spend at these places each month?Open-Ended Response', 'Where else do you shop for food and clothing online, if not Woolworths?',\
                    'Have you used the Woolworths app to shop?', [col for col in features_df.columns if 'Thinking back over the last 3 months, how regularly did you shop with Woolworths for different products' in col], \
                    [col for col in channel_spend_sale_df.columns if 'Over the last 3 months, which method best describes how you most often shop at Woolworths?' in col]
    # create average spend per customer in Woolworths
    cust_avg_spend = channel_spend_sale_df.groupby('Respondent ID').mean()[['Spend at Woolworths']]
    mean_cust_avg_spend = cust_avg_spend['Spend at Woolworths'].mean()
    cust_avg_spend['Spend at Woolworths Untreated'] = cust_avg_spend['Spend at Woolworths'].copy()
    # fill null values with the total average
    cust_avg_spend['Spend at Woolworths'] = cust_avg_spend['Spend at Woolworths'].fillna(mean_cust_avg_spend)
    # create categorical ranges for the spend at Woolworths
    untreated_avg_spend = cust_avg_spend['Spend at Woolworths Untreated']
    treated_avg_spend = cust_avg_spend['Spend at Woolworths']
    spend_cat_list = ['R350-R500', 'R500-R1000', 'R1000-R2000', 'Above R2000']
    untreated_spend_condition = [untreated_avg_spend<500, untreated_avg_spend<1000,
                    untreated_avg_spend<2000, untreated_avg_spend>=2000]
    treated_spend_condition = [treated_avg_spend<500, treated_avg_spend<1000,
                    treated_avg_spend<2000, treated_avg_spend>=2000]
    cust_avg_spend['Spend at Woolworths Untreated Range'] = np.select(
        untreated_spend_condition, spend_cat_list
    )
    cust_avg_spend['Spend at Woolworths Range'] = np.select(
        treated_spend_condition, spend_cat_list
    )
    cust_avg_spend['Spend at Woolworths Untreated Range'] = pd.Categorical(
        cust_avg_spend['Spend at Woolworths Untreated Range'], ordered=True, categories=spend_cat_list
    )
    cust_avg_spend['Spend at Woolworths Range'] = pd.Categorical(
        cust_avg_spend['Spend at Woolworths Range'], ordered=True, categories=spend_cat_list
    )
    # merge the spend columns with the original df
    customer_df = customer_df.merge(cust_avg_spend.reset_index(), on='Respondent ID', how='left')
    # create a spend ratio column of spend at woolworths vs spend at competitors
    customer_df['spend_ratio_ww_vs_comp'] = customer_df['Spend at Woolworths'] / customer_df['competitor_spend'].map(lambda x:1 if x==0 else x)
    
    # NOTE: Purchase behaviour modelling
    # create dataframes for numeric frequency, income, and spend
    freq_income = customer_df[['general_freq', 'income']]
    ww_spend = np.log(customer_df[['Spend at Woolworths']])
    
    # create raw and features dataframe
    bhv_df_raw = customer_df[['general_freq', 'income', 'Spend at Woolworths']]
    bhv_df_features = pd.concat([freq_income,ww_spend,],axis=1)
    
    # create dataframe of interpreted cluster results
    bhv_cluster_result = pd.DataFrame(data={
        'Cluster':['0','1','2','3'],
        'Category':['low frequency, high income, mid spend', 'low frequency, low income, low spend', 'low frequency, low income, mid spend', 'high frequency, mid income, mid spend'],
        'Persona':['prime target for upselling potential','low value customer','price sensitive customer', 'loyal customer with upselling potential']
    })
    
    # create standardized df and sses
    bhv_df_norm, bhv_sse = udfs.kmeans(bhv_df_features)
    
    # create clusters and merge the results
    bhv_df_raw, bhv_df_norm, bhv_df_melt_chart, bhv_df_relative_imp_chart = udfs.kmeans_cust_seg(bhv_df_raw, bhv_df_norm,n_clusters=len(bhv_cluster_result))
    bhv_df_raw = bhv_df_raw.merge(bhv_cluster_result, how='left', on='Cluster')
    
    # add cluster result columns to the original df
    customer_df = customer_df.assign(purchase_behaviour_cluster = bhv_df_raw['Cluster'],
                                    FIS_category = bhv_df_raw['Category'],
                                    FIS_persona = bhv_df_raw['Persona'])
        
    # NOTE: Competitor landscape/loyalty
    # create raw and features df
    lty1_df_raw = customer_df[['Respondent ID', 'general_freq','Number of competitors','spend_ratio_ww_vs_comp']]
    freq = customer_df[['general_freq']]
    num_comp = np.cbrt(customer_df[['Number of competitors']])
    spend_ratio_ww_vs_comp = np.log(customer_df[['spend_ratio_ww_vs_comp']])
    lty1_df_features = pd.concat([freq,num_comp,spend_ratio_ww_vs_comp],axis=1)
    
    # separate customers with spend ratio
    lty1_df_fns_raw = lty1_df_raw.dropna(how='any')
    lty1_df_fns_features = lty1_df_features.dropna(how='any')
    # create df of interpreted cluster results
    lty1_cluster_result = pd.DataFrame(data={
        'Cluster':['0','1','2','3','4'],
        'FNS_category':['low frequency, mid competition, low spend ratio', 'high frequency, high competition, low to mid spend ratio', 'low frequency, high competition', 'low frequency, low number of competitors, high spend ratio', 'high frequency, low competition, high spend ratio'],
        'FNS_persona':['has a few other alternatives', 'prefers Woolworths but can still be swayed by competitors', 'has many alternatives and little loyalty to Woolworths', 'prefers Woolworths and has upselling opportunities', 'is loyal to Woolworths']
    })
    
    # standardize df and create sses
    lty1_df_fns_norm, lty1_df_fns_sse = udfs.kmeans(lty1_df_fns_features)
    
    # create clusters and merge with cluster results
    lty1_df_fns_raw, lty1_df_fns_norm, lty1_df_fns_melt_chart, lty1_df_fns_relative_imp_chart = udfs.kmeans_cust_seg(lty1_df_fns_raw, lty1_df_fns_norm, n_clusters=len(lty1_cluster_result))
    lty1_df_fns_raw = lty1_df_fns_raw.merge(lty1_cluster_result, how='left', on='Cluster')
    
    # filter to treated customers
    fns_customer_df = customer_df[customer_df['spend_ratio_ww_vs_comp'].notnull()]
    fns_customer_df = fns_customer_df.merge(lty1_df_fns_raw[['Respondent ID'] + list(lty1_cluster_result.columns)],how='outer',on='Respondent ID')
    fns_customer_df.rename(columns={'Cluster':'loyalty1_cluster'},inplace=True)
    
    # separate customers without spend ratio
    lty2_df_fs_raw = lty1_df_raw[lty1_df_raw['spend_ratio_ww_vs_comp'].isna()].dropna(axis=1)
    lty2_df_fs_features = lty1_df_features[lty1_df_features['spend_ratio_ww_vs_comp'].isna()].dropna(axis=1)
    
    # create df of interpreted cluster results
    lty2_cluster_result = pd.DataFrame(data={
        'Cluster':['0','1','2'],
        'FN_category':['high frequency', 'low frequency, low competition', 'low frequency, high competition'],
        'FN_persona':['is loyal to Woolworths', 'has apathy towards both Woolworths and competitors', 'has many alternatives and little loyalty to Woolworths']
    })
    
    # create standardized df and sse
    lty2_df_fs_norm, lty2_df_fs_sse = udfs.kmeans(lty2_df_fs_features)
    
    # create clusters and merge cluster results
    lty2_df_fs_raw, lty2_df_fs_norm, lty2_df_fs_melt_chart, lty2_df_fs_relative_imp_chart = udfs.kmeans_cust_seg(lty2_df_fs_raw, lty2_df_fs_norm, n_clusters=len(lty2_cluster_result))
    lty2_df_fs_raw = lty2_df_fs_raw.merge(lty2_cluster_result, how='left', on='Cluster')
    
    # filter to treated customers
    fs_customer_df = customer_df[customer_df['spend_ratio_ww_vs_comp'].isnull()]
    fs_customer_df = fs_customer_df.merge(lty2_df_fs_raw[['Respondent ID'] + list(lty2_cluster_result.columns)],how='outer',on='Respondent ID')
    fs_customer_df.rename(columns={'Cluster':'loyalty2_cluster'},inplace=True)
    
    # concat both treated customers df to recreate the original df
    customer_df = pd.concat([fns_customer_df,fs_customer_df],axis=0)
    
    # combine personas and profiles of both fns and fs
    customer_df['FNs_persona'] = customer_df[['FNS_persona','FN_persona']].bfill(axis=1).iloc[:,0]
    customer_df['FNs_category'] = customer_df[['FNS_category','FN_category']].bfill(axis=1).iloc[:,0]

    # NOTE: Product Preference Segmentation
    # create raw and features df
    product_df_raw = customer_df.loc[:,[col for col in customer_df.columns if 'Shopping Frequency:' in col]]
    for col in product_df_raw.columns:
        product_df_raw[col] = product_df_raw[col].astype(int)
        product_df_raw.rename(columns={col:col.split(':')[1]+' freq'}, inplace=True)
    product_df_features = product_df_raw.copy()
    
    # create df of interpreted cluster results
    product_cluster_result = pd.DataFrame(data={
        'Cluster':['0','1','2','3','4'],
        'Category':['prefers food and buys frequently', 'prefers food and buys it sometimes', 'premium fashion customer, also includes food, and beauty and health products','rarely buys any products','premium home product customer, also sometimes includes fashion products'],
        'Persona':['a prime target for food promotions','an okay target for food promotions', 'a prime target for fashion loyalty programs with opportunities in food, and beauty and health products','not particularly keen for any product','a prime target for home product loyalty programs with opportunities in fashion']
        })
    
    # standardise df and create sses
    product_df_norm, product_df_sse = udfs.kmeans(product_df_features)
    
    # create clusters and merge with cluster results
    product_df_raw, product_df_norm, product_df_melt_chart, product_df_relative_imp_chart = udfs.kmeans_cust_seg(product_df_raw, product_df_norm, n_clusters=len(product_cluster_result))
    product_df_raw = product_df_raw.merge(product_cluster_result, how='left', on='Cluster')
    
    # add the persona and category columns to the original df
    customer_df = customer_df.assign(product_preference_cluster = product_df_raw['Cluster'],
                                    PF_category = product_df_raw['Category'],
                                    PF_persona = product_df_raw['Persona'])


    # NOTE: Channel Preference Segmentation
    # create raw df; the raw df was also used as features df
    channel_selected_cols = [col for col in customer_df.columns if 'Over the last 3 months, which method best describes how you most often shop at Woolworths?' in col]
    selection_list = [col.split('?')[1].split('-')[0] for col in channel_selected_cols]
    channel_df_raw = customer_df.loc[:,channel_selected_cols]
    for col in channel_df_raw.columns:
        channel_df_raw[col] = channel_df_raw[col].fillna(0).map(lambda x:1 if x!=0 else 0)
        if 'Not Applicable' in col:
            channel_df_raw.drop(col,axis=1,inplace=True)
        channel_df_raw.rename(columns={col:col.split('?')[1]}, inplace=True)
    products_list = ['Food', 'Fashion', 'Home', 'Beauty']
    products_df_list, products_df_sse = ({} for _ in range(2))
    for product in products_list:
        products_df = channel_df_raw.loc[:,[col for col in channel_df_raw.columns if product in col]]
        products_df_list[product] = products_df
    
    # create sse
    products_df_sse = udfs.kmodes(products_df_list)
    
    # create clusters
    products_df_list, products_df_melt_chart_dict, products_df_relative_imp_chart_dict = udfs.kmodes_cust_seg(products_df_list, n_clusters=4)        
    
    # interpret clusters for all four product offerings
    channel_clusters = ['0','1','2','3']
    food_cluster_persona = pd.DataFrame({'Cluster':channel_clusters,
                                    'Category':['goes instore and also shops online', 'mostly goes instore but also uses the app frequently', 'mostly uses the app but also goes instore sometimes', 'mostly goes instore'],
                                    'Persona':['has similar preference for instore service and convenience', 'prefers personalized service slightly above convenience', 'prefers convenience','prefers personalized service']})
    fashion_cluster_persona = pd.DataFrame({'Cluster':channel_clusters,
                                    'Category':['goes instore and also shops online', 'mostly shops online', 'rarely uses any of the channels', 'mostly goes instore'],
                                    'Persona':['has similar preference for personalized service and convenience', 'prefers convenience', 'does not particularly prefer any channel','prefers personalized service']})
    home_cluster_persona = pd.DataFrame({'Cluster':channel_clusters,
                                    'Category':['rarely uses any of the channels', 'mostly goes instore but also uses the app sometimes', 'mostly goes instore', 'mostly shops online'],
                                    'Persona':['does not particularly prefer any channel', 'prefers personalized service slightly over convenience', 'prefers personalized service', 'prefers convenience']})
    beauty_cluster_persona = pd.DataFrame({'Cluster':channel_clusters,
                                    'Category':['mostly goes instore', 'mostly uses the app', 'mostly shops online', 'rarely uses any of the channels'],
                                    'Persona':['prefers personalized service', 'prefers convenience', 'prefers convenience', 'does not particularly prefer any channel']})
    channel_cluster_result = {'Food':food_cluster_persona,
                              'Fashion':fashion_cluster_persona,
                              'Beauty':beauty_cluster_persona,
                              'Home':home_cluster_persona}
    
    # merge cluster results with df
    products_df_list['Food'] = products_df_list['Food'].merge(food_cluster_persona,how='left',on='Cluster')
    products_df_list['Fashion'] = products_df_list['Fashion'].merge(fashion_cluster_persona,how='left',on='Cluster')
    products_df_list['Home'] = products_df_list['Home'].merge(home_cluster_persona,how='left',on='Cluster')
    products_df_list['Beauty'] = products_df_list['Beauty'].merge(beauty_cluster_persona,how='left',on='Cluster')
    
    # add the results to the original df
    customer_df = customer_df.assign(food_channel_cluster = products_df_list['Food']['Cluster'],                                        
                                     food_channel_category = products_df_list['Food']['Category'],
                                     food_channel_persona = products_df_list['Food']['Persona'],
                                     fashion_channel_cluster = products_df_list['Fashion']['Cluster'],                                        
                                     fashion_channel_category = products_df_list['Fashion']['Category'],
                                     fashion_channel_persona = products_df_list['Fashion']['Persona'],
                                     home_channel_cluster = products_df_list['Home']['Cluster'],                                        
                                     home_channel_category = products_df_list['Home']['Category'],
                                     home_channel_persona = products_df_list['Home']['Persona'],
                                     beauty_channel_cluster = products_df_list['Beauty']['Cluster'],                                        
                                     beauty_channel_category = products_df_list['Beauty']['Category'],
                                     beauty_channel_persona = products_df_list['Beauty']['Persona'],)
    
    # separate persona columns
    persona_df = customer_df.loc[:,['Respondent ID'] + [col for col in customer_df.columns if 'persona' in col]]
    
    # merge persona columns with dashboard dfs
    features_df = features_df.merge(persona_df, on='Respondent ID', how='left')
    channel_spend_sale_df = channel_spend_sale_df.merge(persona_df, on='Respondent ID', how='left')
    benefits_df = benefits_df.merge(persona_df, on='Respondent ID', how='left')
    competitor_df = competitor_df.merge(persona_df, on='Respondent ID', how='left')         

    return customer_df, features_df, channel_spend_sale_df, benefits_df, competitor_df, id, gender_col, cat_age_col,\
    num_age_col, cat_income_col, num_income_col, shop_for_col, high_freq_col, wwspend_col, saletype_col,\
    benefits_col, comp_num_col, comp_spend_col, comp_list_col, used_app_col, prod_pref_list, channel_pref_list,\
    bhv_df_raw, bhv_df_features, bhv_cluster_result, bhv_sse, bhv_df_melt_chart, bhv_df_relative_imp_chart,\
    lty1_df_fns_raw, lty1_df_fns_features, lty1_cluster_result, lty1_df_fns_sse, lty1_df_fns_melt_chart,\
    lty1_df_fns_relative_imp_chart, lty2_df_fs_raw, lty2_df_fs_features, lty2_cluster_result, lty2_df_fs_sse,\
    lty2_df_fs_melt_chart, lty2_df_fs_relative_imp_chart, fns_customer_df, fs_customer_df, product_df_raw,\
    product_df_features, product_cluster_result, product_df_sse, product_df_melt_chart, product_df_relative_imp_chart,\
    products_df_sse, products_df_melt_chart_dict, products_df_relative_imp_chart_dict, products_df_list,\
    channel_selected_cols, channel_clusters, channel_cluster_result

@st.cache_data(ttl=10*60)
def save_model(customer_df):
    gc = gs.authorize(gs_credentials) # give gspread library access to the private google sheet
    gs_db = gc.open_by_url(st.secrets["private_gsheets_url"]) # open the spreadsheet
    sheet_to_update = gs_db.worksheet('Sheet1') # open the specific sheet to update
    customer_df = customer_df.astype(str) # convert all columns to strings to bypass any exceptions that may occur
    customer_df.fillna('',inplace=True) # fill null values
    # update the sheet
    try: sheet_to_update.update([customer_df.columns.values.tolist()] + customer_df.values.tolist(), value_input_option='USER_ENTERED')
    except: st.write('Segmentation model output was not saved to google sheets')