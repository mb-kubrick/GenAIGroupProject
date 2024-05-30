import streamlit as st
import sqlite3
import pandas as pd

import sqlite3


st.title("Allocations")
db_url = "portfolio_allocations.db"
db3_url = "portfolio_allocations3.db"
db5_url = "portfolio_allocations5.db"
db10_url = "portfolio_allocations10.db"

conn = sqlite3.connect(db_url)
conn3 = sqlite3.connect(db3_url)
conn5 = sqlite3.connect(db5_url)
conn10 = sqlite3.connect(db10_url)

#to dataframe
def run_query(query, conn):
    return pd.read_sql_query(query, conn)

query = "SELECT * FROM portfolio"
query3 = "SELECT * FROM portfolio3"
query5 = "SELECT * FROM portfolio5"
query10 = "SELECT * FROM portfolio10"

try:
    data = run_query(query, conn)
    data3 = run_query(query3, conn3)
    data5 = run_query(query5, conn5)
    data10 = run_query(query10, conn10)
    st.write(data)
    st.write(data3)
    st.write(data5)
    st.write(data10)
except Exception as e:
    st.error(f"An error occurred: {e}")

conn.close()
