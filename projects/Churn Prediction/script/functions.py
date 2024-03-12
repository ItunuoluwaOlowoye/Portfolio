from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from typing import Union
import streamlit as st
import seaborn as sns
from PIL import Image
import altair as alt
import pandas as pd
import numpy as np
import os

############### DEFAULT VARIABLES ####################

logo = Image.open('pictures/logo.png')
primary_color = '#2F0A0D'
text_color = '#524748'
secondary_color = '#3E0E12'
tertiary_color = '#7B6F71'
grey_color = '#E7E5E5'
store_name = 'Telco'
color_domain = ['Yes', 'No']
color_range = [secondary_color, grey_color]
string_columns = ['Gender', 'InternetService', 'Contract', 'PaymentMethod']
category_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'PaperlessBilling', 'Churn']
integer_columns = ['Tenure']
float_columns = ['MonthlyCharges', 'TotalCharges']
string_mapping = {0:'No', 1:'Yes'}
numeric_mapping = {'No':0, 'Yes':1}
other_mapping = {'No phone service':'No', 'No internet service':'No'}
model_svm = LinearSVC(max_iter=50000, dual=True)
model_svm = CalibratedClassifierCV(model_svm)
models = {'Logistic Regression':LogisticRegression(), 'Support Vector Machines (SVM)':model_svm,
          'Random Forest Classifier':RandomForestClassifier(), 'Gaussian Naive Bayes':GaussianNB(),
          'K-Nearest Neighbors (KNN)':KNeighborsClassifier(),}


############## UI FUNCTIONS ##########################

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

# create a new container
def new_container(command:st, border:bool=True, height:Union[int, None]=None):
    return command.container(border=border, height=height)

# customer profiles intro columns
def display_intro_columns():
    # create content
    churn_insights = f"""{store_name} is a telecommunications company that provides home phone service and
    internet subscriptions. We first inspect and explore the data to understand the factors that affected
    churn in the last quarter"""
    churn_model = """Churn means customers are leaving the network, which means revenue loss. The
    essence of training a churn prediction model is to avoid losing too much revenue"""
    actionable_insights = """Features of customers predicted to churn are studied to offer them incentives and
    targeted marketing campaigns based on their interests to mitigate against churn"""

    # Create columns for content
    cols = st.columns(3)
    # for each section, create contact card with icon, title, body, and action button
    for col, icon, title, content, button_text in zip(
        cols,
        ("💡", "🌀", "🎯"),
        ("Analysis Insights", "Churn Model", "Recommended Actions"),
        (churn_insights, churn_model, actionable_insights),
        ("Investigate Further", "Training and Testing", "View Actions"),
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

# faceted bar chart
def facet_bar_chart(df:pd.DataFrame, xaxis:str, yaxis:str, facet:str, color_domain:list=color_domain,
                    color_range:list=color_range, text_color:str=text_color):
    # Create the bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(xaxis).axis(title=None),
        y=alt.Y(yaxis).axis(title=None),
        color=alt.Color(xaxis).legend(None).scale(domain=color_domain, range=color_range),
        text=alt.Text(yaxis).format('.0%')
    ).properties(height=100, width=100)
    text = chart.mark_text(align='center', dy=-5, color=text_color)
    chart = alt.layer(chart, text).facet(facet=alt.Facet(facet).title(None), columns=2)
    return chart

# 100% stacked bar chart
def stacked_bar_chart(df:pd.DataFrame, column:str, hue:str='Churn', color_domain:str=color_domain,
                      color_range:str=color_range, chart_height:int=200):
    stack_df = df.groupby([column, hue]).size()
    stack_df_sum = df.groupby(column).size()
    stack_df = (stack_df/stack_df_sum).reset_index(name='percentage')
    
    chart = alt.Chart(stack_df).mark_bar().encode(
        x=alt.X('percentage').axis(title=None).stack('normalize'),
        y=alt.Y(column).axis(labelLimit=180),
        color=alt.Color(hue).legend(orient='top', labelFontSize=12)
        .scale(domain=color_domain, range=color_range),
        order=alt.Order(hue).sort('descending')
    ).properties(height=chart_height)
    return chart

# analysis ingishts
@st.cache_data(ttl="1h")
def analysis_insights(df:pd.DataFrame):
    st.markdown('<span style="font-weight:bold; font-size:24px">Analysis Insights</span>',
                unsafe_allow_html=True)
    # calculate churn rate
    churn_rate = df['Churn'].value_counts(normalize=True)
    # calculate acquisition rate
    joined_last_quarter = df['Tenure'] <=3
    acquisition_rate = joined_last_quarter.map({True:'Yes', False:'No'}).value_counts(normalize=True)
    # compare churn and acquisition in the last quarter
    churn_vs_acquisition = pd.concat([churn_rate, acquisition_rate], axis=0,
                                    keys=['Churn','Acquisition']).reset_index()
    
    # create columns for insights
    bar_scatter_col, heatmap_col = st.columns([1,2])
    
    st.divider()

    col1, col2, col3 = st.columns(3, gap='medium')

    # Create the churn vs acquisition chart
    chart = facet_bar_chart(df=churn_vs_acquisition, xaxis='Churn', yaxis='proportion', facet='level_0')
    with new_container(bar_scatter_col):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    The <strong>churn rate is higher than the acquisition rate</strong> in the last quarter,
                    thus, churn rate is too high and needs to be mitigated.
                    </div>''',
                    unsafe_allow_html=True)
    
    scatter_df = telco_df.copy()
    scatter_df['Churn'] = scatter_df['Churn'].replace(numeric_mapping).astype(float)
    scatter_df = scatter_df.groupby('Tenure', as_index=False).Churn.mean()
    chart = alt.Chart(scatter_df).mark_point(color=secondary_color).encode(
        x='Tenure', y='Churn'
    )
    with new_container(bar_scatter_col):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    <strong>More recent customers are most likely to churn</strong> so {store_name} should be
                    particularly
                    interested in new customers</div>''',
                    unsafe_allow_html=True)
    
    chart = stacked_bar_chart(df=df, column='SeniorCitizen')
    with new_container(col1):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Senior citizens are more likely to churn. This might be from decreased phone usage
                    due to old age</div>''',
                    unsafe_allow_html=True)

    chart = stacked_bar_chart(df=df, column='Partner')
    with new_container(col1):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Customers without partners are more likely to churn</div>''',
                    unsafe_allow_html=True)

    chart = stacked_bar_chart(df=df, column='StreamingTV')
    with new_container(col1):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Customers who stream TV are more likely to churn. Perhaps it is overpriced?</div>''',
                    unsafe_allow_html=True)
    
    chart = stacked_bar_chart(df=df, column='StreamingMovies')
    with new_container(col2):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Customers who stream movies are also more likely to churn. Perhaps it is overpriced?</div>''',
                    unsafe_allow_html=True)
    
    chart = stacked_bar_chart(df=df, column='PaperlessBilling')
    with new_container(col2):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Customers whose billing method is paperless are more likely to churn</div>''',
                    unsafe_allow_html=True)
    
    chart = stacked_bar_chart(df=df, column='InternetService')
    with new_container(col2):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Customers who use fiber optics <span style='font-size:12px'> (highly correlated with
                    monthly charges)</span> are more likely to churn.</div>''',
                    unsafe_allow_html=True)
    
    chart = stacked_bar_chart(df=df, column='Contract')
    with new_container(col3):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    Customers with month-to-month contracts are less committed and more likely to churn
                    </div>''',
                    unsafe_allow_html=True)
    
    chart = stacked_bar_chart(df=df, column='PaymentMethod')
    with new_container(col3):
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    People who pay with electronic checks are more likely to churn. Is the UX
                    a tough ordeal?</div>''',
                    unsafe_allow_html=True)
    
    df = data_eng(df=df)
    corr_df = df[['StreamingTV', 'StreamingMovies', 'MonthlyCharges',
                  'InternetService_Fiber optic',]].corr().round(2)
    fig, ax = plt.subplots(figsize=(16, 9))
    mask = np.zeros_like(corr_df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Setup the correlation matrix as a heatmap with diverging colors
    sns.heatmap(corr_df, mask=mask, cmap='seismic', annot=True, vmax=1, vmin=-1, linewidths=.5)
    with new_container(heatmap_col, height=770):
        st.pyplot(fig)
        st.markdown(f'''<div style='font-size:14px; padding-bottom:5px'>
                    <p>There is a high positive correlation between fibre optic service and monthly charges.
                    In subsequent analyses, we see that fiber optic users are more likely to churn. <strong>A
                    hypothesis is that the fiber optic service is overpriced and {store_name} should
                    revisit the pricing strategy.</strong></p>
                    <p>There is also some relationship between streaming services (TV and movies) and
                    monthly charges. In subsequent analyses, we also see that customers who use streaming services
                    are slightly more prone to churn than others. While {store_name} does not charge additional
                    fees for streaming, <strong>limited-time streaming bundles can be offered as incentives to keep
                    customers streaming within the network.</strong></p>
                    </div>''',
                    unsafe_allow_html=True)

# churn model
def churn_model(df:pd.DataFrame):
    # randomise data
    df = df.sample(frac=1, random_state=42)
    # select important features
    X = df.loc[:, ['SeniorCitizen', 'Partner', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                   'InternetService_Fiber optic', 'Contract_One year', 'Contract_Two year',
                   'PaymentMethod_Electronic check', 'Tenure', 'Churn']]
    
    # create training group
    train_grp = X.groupby('Churn').sample(n=1600,random_state=42)
    # create test group
    test_grp = X.drop(train_grp.index)
    
    X_train = train_grp.drop('Churn',axis=1)
    y_train = train_grp['Churn']
    X_test = test_grp.drop('Churn',axis=1)
    y_test = test_grp['Churn']

    selection = st.radio(label='Classification Model', horizontal=True,
                         options=['Logistic Regression', 'Support Vector Machines (SVM)',
                                  'Random Forest Classifier', 'K-Nearest Neighbors (KNN)',
                                  'Gaussian Naive Bayes'])
    
    st.divider()
    
    # instantiate and fit a Logistic Regression model
    model = models[selection]
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    cr = classification_report(y_true=y_test, y_pred=y_predict)
    cm = confusion_matrix(y_true=y_test, y_pred=y_predict)
    accuracy, precision_retained, precision_churn, recall_retained, recall_churn =\
        precision_recall_calc(model_confusion_matrix=cm)
    
    # AUC score
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # ROC curve
    fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=1)

    df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate':tpr})

    chart = alt.Chart(df).mark_line(color=secondary_color).encode(
        x='False Positive Rate', y='True Positive Rate'
    )

    col1, col2, col3 = st.columns(3)
    with new_container(col1, height=400):
        st.markdown(f'The Area Under Curve (**AUC**) score is **{auc_score*100:.2f}%**')
        st.altair_chart(chart, use_container_width=True)
    new_container(col2).metric(label='Accuracy', value=f'{accuracy*100:.2f}%')
    new_container(col2).metric(label='Precision of retained customers prediction',
                               value=f'{precision_retained*100:.2f}%')
    new_container(col2).metric(label='Precision of churned customers prediction',
                               value=f'{precision_churn*100:.2f}%')
    new_container(col3).metric(label='Recall value of retained customers prediction',
                               value=f'{recall_retained*100:.2f}%')
    new_container(col3).metric(label='Recall value of churned customers prediction',
                               value=f'{recall_churn*100:.2f}%')
    new_container(col3).metric(label='Accuracy', value=f'{accuracy*100:.2f}%')


############## DATA FUNCTIONS #############

def data_wrangling(file:str='wa_telco.csv'):
    # load data into a dataframe
    telco_df = pd.read_csv(file)
    telco_df.rename(columns={'customerID':'CustomerID', 'gender':'Gender', 'tenure':'Tenure'}, inplace=True)
    telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce', downcast='integer')
    # fill null total charges with monthly charges
    telco_df.loc[telco_df['TotalCharges'].isna(), 'TotalCharges'] = telco_df.loc[telco_df['TotalCharges'].isna(),
                                                                                'MonthlyCharges']
    for col in category_columns:
        telco_df[col] = pd.Categorical(telco_df[col])
        telco_df[col] = telco_df[col].replace(other_mapping).replace(string_mapping)
    telco_df.set_index('CustomerID', inplace=True)
    return telco_df

def data_eng(df:pd.DataFrame):
    # Generating the dummy variables
    for col in df.columns:
        try: df[col] = df[col].replace(numeric_mapping).astype(int)
        except: pass
    df = pd.get_dummies(data=df, columns=string_columns, drop_first=True)
    return df

# calculate accuracy, precision, and call
def precision_recall_calc(model_confusion_matrix):
    tn = model_confusion_matrix[0,0]
    fp = model_confusion_matrix[0,1]
    fn = model_confusion_matrix[1,0]
    tp = model_confusion_matrix[1,1]
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    precision_retained = tn/(fn+tn)
    precision_churn = tp/(fp+tp)
    recall_retained = tn/(tn+fp)
    recall_churn = tp/(fn+tp)
    
    return accuracy, precision_retained, precision_churn, recall_retained, recall_churn

telco_df = data_wrangling()