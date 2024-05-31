import streamlit as st
from agent import call_agent
from ml_flow import mlflow_server, evaluate_agent, get_info_on_runs, delete_all_runs

query = 'What is the best stock to invest into?'

response, agent = call_agent(query)

server_process = mlflow_server()
evaluate_agent(agent, query, 'mlflow_agent')

txt, metrics = get_info_on_runs('mlflow_agent_development')
st.write(txt)

delete_all_runs('mlflow_agent_development')