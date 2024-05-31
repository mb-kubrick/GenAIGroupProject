"""File for running the project pipeline form start to finish. This includes:
1. Downloading 10K reports from the SEC website.
2. Cleaning the 10K files.
3. Embedding the 10K files.
4. Storing the embeddings in a Vector Database (Milvus).
5. Generating random allocations of stocks so as to generate a synthetic client database.
6. Supplying an Agent LLM model which can act as a broker.
7. Passing models to MLFlow.
"""

# IMPORTS & SETUP ------------------------------------------------------------------------------------------------------

import os
import logging

import pandas as pd
import random
import sqlite3
import yfinance as yf

from data_access import download_files_10k
from langchain.agents import AgentExecutor
from data_processing import write_clean_html_text_files
from ml_flow import mlflow_server, evaluate_llm
from vector_database import start_docker_compose, start_attu_container, create_milvus_db
from agent_presentation import call_agent

logging.basicConfig(level=logging.INFO)

# DEFINING FUNCTIONS ---------------------------------------------------------------------------------------------------

def run_10ks(ticker, dest_folder, name_folder_txt):
    download_files_10k(ticker, dest_folder)
    write_clean_html_text_files(dest_folder, name_folder_txt)

def run_vector_database():
    start_docker_compose()
    start_attu_container()
    collection = create_milvus_db()

    return collection

def run_generate_synthetic_data():
    db_conn = sqlite3.connect('portfolio_allocationsTest.db') 
    db_cur = db_conn.cursor()
    engine = create_engine('sqlite:///portfolio_allocationsTest.db').connect()
    create_synthetic_data(db_conn, db_cur)
    get_share_value(db_conn, db_cur, engine)
    pass

def run_agent_model(collection):
    pass

def run_mlflow(agent: AgentExecutor, experiment_name: str = 'mlflow_development', delete: bool = False) -> None:
    """Runs the MLFlow portion of the pipeline.

    Args:
        agent (AgentExecutor): The agent with which to perform the evaluation.
        experiment_name (str, optional): The name of the MLFlow experiment. Defaults to 'mlflow_development'.
        delete (bool, optional): Whether to delete runs after completion. Defaults to False.
    """
    _ = mlflow_server()

    eval_set = pd.read_csv(os.getcwd() +'/data/' + 'Evaluation Dataset - Agent.csv')
    evaluate_agent(agent, eval_set['question'], experiment_name)
    text, metrics = get_info_on_runs(experiment_name)

    print(text)

    if delete:
        delete_all_runs(experiment_name)

# RUNNING --------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_10ks('AAPL', './demo_data/AAPL_html_files', './demo_data/AAPL_cleaned_txt_files')
    collection = run_vector_database()
    run_generate_synthetic_data()
    agent = run_agent_model(collection)
    run_mlflow(agent)

    
