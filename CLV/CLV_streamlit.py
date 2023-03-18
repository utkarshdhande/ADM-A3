import pandas as pd
from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import call_udf
import streamlit as st
import snowflake.connector as sf
import json

# function to create Snowflake session

def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("creds.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

# Streamlit UI
st.title('Customer Life Time Value Prediction')

st.text('')
csv_file = st.file_uploader(label='Upload CSV file with user data', type = 'csv')

# function to calculate CLV
def predict(inputs):
    # create Snowflake session
    # session = create_session()
    # connection = session.connect()
    # cursor = session.cursor()

    Streamlit_data = session.table('Streamlit_data')

    # query the temporary table to calculate CLV
    snowdf_results = Streamlit_data.select(*inputs,
                        call_udf("clv_xgboost_udf").alias('PREDICTION'), 
                        (F.col('TOTAL_SALES')).alias('ACTUAL_SALES')
                        )
        # results = cursor.fetchall()
    # cursor.close()
    # connection.close()

    # convert results to Pandas DataFrame and return
    # return pd.DataFrame(results, columns=['Customer ID', 'CLV'])

# Display the contents of the uploaded file as a DataFrame
if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        st.write(df)
        session = create_session()
        session.use_warehouse('FE_AND_INFERENCE_WH')
        session.use_database('tpcds_xgboost')
        session.use_schema('demo')
        df = session.create_dataframe(df)
        prediction = predict(df).topandas()
        st.write(prediction)
    except Exception as e:
        st.write("Error reading CSV file:", e)
