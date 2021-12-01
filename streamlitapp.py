import streamlit as st
import pandas as pd
import altair as alt
from dm_tools import data_prep
from model import model_train,model_compare

from urllib.error import URLError

@st.cache
def get_data():
    # AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv("veteran.csv")
    return df


code = """
def data_prep():
    # read the veteran dataset
    df = pd.read_csv('veteran.csv')
    
    # change DemCluster from interval/integer to nominal/str
    df['DemCluster'] = df['DemCluster'].astype(str)
    
    # change DemHomeOwner into binary 0/1 variable
    dem_home_owner_map = {'U':0, 'H': 1}
    df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)
    
    # denote errorneous values in DemMidIncome
    mask = df['DemMedIncome'] < 1
    df.loc[mask, 'DemMedIncome'] = np.nan
    
    # impute missing values in DemAge with its mean
    df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

    # impute med income using mean
    df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

    # impute gift avg card 36 using mean
    df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)
    
    # drop ID and the unused target variable
    df.drop(['ID', 'TargetD'], axis=1, inplace=True)
    
    # one-hot encoding
    df = pd.get_dummies(df)
    
    return df
"""

st.write("### Background describe")
st.write("""
Business Scenario: A national veterans organisation is seeking to improve their donation solicitations by
targeting the potential donors. By only focusing on the supposedly donors, less money can be spent on
solicitation efforts and more money can be available for charitable endeavors. Of particular interest is the class
of individuals identified as lapsing donors. They have ran a greeting card mailing campaign called Veteran. The
organisation now seeks to classify its lapsing donors based on their responses to this campaign. With this
classification, a decision can be made to either solicit or ignore a lapsing individual in the next year campaign.
as a data science professional, to use this dataset to understand the patterns of donation and identify
supposedly donors to improve the solicitation effort.
The Veteran dataset is available as veteran.csv file. Before we build the predictive models, we will
examine the dataset to understand its basic characteristics such as data dimensions, attributes nature, data
distribution, outliers, data quality etc.
""")


df = get_data()
cat_list = []
for i in df.columns:
    if df.loc[:,i].dtypes == "int64" or df.loc[:,i].dtypes == "object":
        # print(i)
        cat_list.append(i)
ctv = st.selectbox(
    "Choose category variable to change plot investigate the distribute of categorical data", cat_list
)
# st.write('You selected:', ctv)
if not ctv:
    st.error("wrong")
else:
    data = df
    # data /= 1000000.0
    st.write("### The Original Data", data)

    # data = data.T.reset_index()
    # data = pd.melt(data, id_vars=["index"]).rename(
    #     columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
    # )
    st.write("##### Category data explore")
    chart = (
        alt.Chart(df).mark_bar()
        .encode(
            x=ctv,
            y="count()",
            # color = "mean(Streams)",
        )
    )
    st.altair_chart(chart)

    df_pre = data_prep()
    # st.write("### Background describe")
    st.write('### Preprocessed Data',df_pre)
    st.code(code)
    st.write("""
        The preprocess steps follow above code, the comments shows the detail process such as transfer categorical data
        to numerical data, drop the Nan values and drop the useless features.
        """)



    # values = st.slider('Select a range of values',0.0, 100.0, (25.0, 75.0))
    # st.write('Values:', values)
    st.write("### Model training")

    param = st.select_slider('### Select the lambda value for LR model',options = [0.001,0.01,0.1,1.0])
    st.write("we can try to change the C param for LR model and investigate the result changes, so we can find the how the param changes influerence resuilts. ")

    if st.button('Train LR model'):
        cr,trains,tests = model_train(df_pre,param)
        st.write("### Show Result")
        st.write("Training dataset accuracy score:",trains)
        st.write("Test dataset accuracy score:", tests)
        st.write("Classification Report:\n")
        st.write(cr)
        # st.write('Why hello there')
        st.write('we can change the parameter to compare the result of LR model that find a most suitable parameter.')
    else:
        st.write('Wait Training')


    if st.button('Compare models with default parameters'):
        df_r = model_compare(df_pre)
        st.write("### Show Result")
        st.write(df_r)
        st.write("we can see that NaiveBayes and LR model get the faster speed, DecesionTree Model get heavy overfit, "
                 "GBDT model get the best test score and robust result, but cost more time. if we chase speed, we can choose"
                 "LR or NaiveBayes, if we chase the model accuracy and robust, we can choose GBDT model.")
    else:
        st.write("Waiting...")




    # st.write('My favorite color is', param)



if __name__ == '__main__':
    # df = get_data()
    # # print(df.info())
    # cat_list = []
    # for i in df.columns:
    #     if df.loc[:, i].dtypes == "int64" or df.loc[:, i].dtypes == "object":
    #         cat_list.append(i)
    # print(cat_list)
    pass