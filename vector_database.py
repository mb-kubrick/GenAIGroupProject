from dotenv import load_dotenv
import os
import subprocess
import docker
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from milvus import default_server
import numpy as np
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus # Can use coher/anthropic etc
from langchain_community.document_loaders import TextLoader

from sentence_transformers import SentenceTransformer
import pickle
import re
import pandas as pd


def start_docker_compose():
    try:
        # Run the docker-compose up command
        result = subprocess.run(['docker', 'compose', 'up', '-d'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Docker Compose started.")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        # print("Failed to start Docker Compose.")
        # print(e.stderr.decode())
        # raise
        pass
    else:
        raise


def start_attu_container():
    """This function first checks if you have the zilliz docker container with name
    'attu_container' inside an image called 'zilliz/attu:v2.3.9' already running.
    If it exists it returns this container, otherwise it creates and returns it.
    NOTE: Make sure you have your Docker daemon running.

    Returns:
        container (Container): Your zilliz docker container.
    """
    client = docker.from_env()

    # Check if any container with the specified image is already running
    container = None
    for cont in client.containers.list(all=True):
        if 'zilliz/attu:v2.3.9' in cont.image.tags:
            container = cont
            break
    
    # Check if the container is already running
    try:
        container = client.containers.get('attu_container')
        if container.status != 'running':
            container.start()
            print("Attu container started.")
        else:
            print("Attu container is already running.")
    except docker.errors.NotFound:
        # Container is not found, so we create and start it
        container = client.containers.run(
            "zilliz/attu:v2.3.9",
            name="attu_container",
            ports={'3000/tcp': 8000},
            environment={"MILVUS_URL": "0.0.0.0:19530"},
            detach=True
        )
        print("Attu container created and started.")
    return container


def load_10k_data():
    """This function loads in the txt files containing the 10K annual reports from
    the /data/txt_files folder. It loads them in using LangChains TextLoader and adds
    these all into a list.

    Returns:
        docs (list): A list of LangChain Documents for each of the 10K txt files
    """
    # Get all file paths to 10K txt reports
    data_dir_path =  os.getcwd() + '/data/txt_files'
    txt_dirs = os.listdir(data_dir_path)
    docs = []

    for txt_dir in txt_dirs:
        txt_dir_path = os.path.join(data_dir_path, txt_dir)
        path_generator = Path(txt_dir_path).glob('*.txt')
        path_list = [str(path) for path in path_generator]

        # Load txt files as LangChain Documents
        for file_path in path_list:
            loader = TextLoader(file_path)
            content = loader.load()
            docs += content

    return docs


def create_milvus_connection():
    """This function assumes you have your docker containers set up and attempts to
    connect to your local Milvus server.

    Returns:
        vector_library (Collection): A Milvus Collection with schema as defined
    """
    # Attempt to connect to milvus db
    # Make sure you have docker and zilliz container running
    try:
        default_server.start()
        connections.connect("default", host="127.0.0.1", port=default_server.listen_port)
    except TimeoutError:
        connections.connect("default", host="localhost", port="19530")
    else:
        raise Exception("Could not connect to Milvus db")

    # Drop collection if it exists
    utility.drop_collection("AnnualReportDataSearch")

    # Create schema for Milvus Collection
    fields = [
        FieldSchema(
        name="pk",
        description="Primary Key",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False,),

        FieldSchema(
        name="embedding",
        description="10K embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=384,
        ),
    ]
    schema = CollectionSchema(fields, description="10K embeddings")
    vector_library = Collection("AnnualReportDataSearch", schema)
    return vector_library


def init_miluvs(vector_library, vector_embeddings):
    """Function to instantiate your Milvus Collection with primary key and vector embeddings.
    It used an IVF Flat index with an L2 metric. If successful, the Milvus db is ready
    to be queried.

    Args:
        vector_library (Collection): Milvus Collection with defined schema
        vector_embeddings (list): list of embeddings for each chunk of data

    Returns:
        (str): 'Success' or 'Failed' depending on if your instantiation worked
    """
    # Attempt to initialise Milvus Collection with given vector embeddings
    try:
        vector_library.release()
        vector_library.drop_index()
        database_insert = [
            [i for i in range(len(vector_embeddings))],  
            vector_embeddings,  
        ]
        vector_library.insert(database_insert)
        index_params = {
            "index_type": "IVF_FLAT", 
            "metric_type": "L2",
            "params": {"nlist": 1}  
        }
        vector_library.create_index("embedding", index_params)
        vector_library.flush()
        vector_library.load()
        return 'Success'
    except Exception as e:
        print(e)
        return 'Failed'


def create_milvus_db():
    """Function to combine all processes required to setup a Milvus db
    with vector embeddings relating to 10K reports in the /data/txt_files folder.
    """
    # Load 10K data as list of LangChain Documents
    docs_10k = load_10k_data()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs_10k)

    # Connect to Milvus db and embed chunks
    vector_library = create_milvus_connection()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_embeddings = []
 
    # Create vector embeddings
    for split in splits:
        embedding = model.encode(split.page_content)
        vector_embeddings.append(embedding)

    # Upload embeddings to Milvus Collection
    init_miluvs(vector_library, vector_embeddings)


if __name__ == '__main__':
    start_docker_compose()
    start_attu_container()
    create_milvus_db()
