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
from data_access import download_files_10k
from ml_flow import mlflow_server, evaluate_llm
from vector_database import start_docker_compose, start_attu_container, create_milvus_db

logging.basicConfig(level=logging.INFO)

# DEFINING FUNCTIONS ---------------------------------------------------------------------------------------------------

def run_10ks(ticker, dest_folder):
    download_files_10k(ticker, dest_folder)

def run_vector_database():
    start_docker_compose()
    start_attu_container()
    create_milvus_db()

    return collection

def run_generate_synthetic_data():
    # get_connections
    # drop_table
    # create_table
    # insert_into_table
    # get_table
    pass

def run_agent_model(collection):
    pass

def run_mlflow(agent_model):

    mlflow_server()

    eval_set = pd.read_csv(os.getcwd() +'/data/' + 'Evaluation Dataset.csv')
    evaluate_llm(agent_model, eval_set, "openai:/gpt-3.5-turbo", "mlflow_development")

# RUNNING --------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run_10ks('AAPL', '.\AAPL_html_files_recent')
    collection = run_vector_database()
    run_generate_synthetic_data()
    run_agent_model(collection)
    run_mlflow()
