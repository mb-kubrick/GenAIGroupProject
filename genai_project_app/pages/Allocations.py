import streamlit as st
import sqlite3
import pandas as pd


import sqlite3


st.title("Allocations")

db_url = "portfolio_allocations3.db"


conn = sqlite3.connect(db_url)

#to dataframe
def run_query(query):
    return pd.read_sql_query(query, conn).set_index("ClientId").drop(columns="index")

query = "SELECT * FROM portfolio3"

try:
    data = run_query(query)
    st.write(data)
except Exception as e:
    st.error(f"An error occurred: {e}")

conn.close()


