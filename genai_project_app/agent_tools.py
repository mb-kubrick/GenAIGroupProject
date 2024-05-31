# Specifying all the imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, ListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.few_shot import FewShotPromptTemplate
import re
import os

load_dotenv()

def get_numbers(query: str):
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0.2)
    template = """
    You are smart agent that can idenify all the client ids to solve the query. 
    If no start point for a client is mentioned, assume client 1 is the first client.
    id. From the query, return all the ids that the query refers to as a list of integers.
    {format_instructions}
    {query}
    """

    class Numbers(BaseModel):
        numbers: str = Field(description="should say 'the numbers extracted from the query'")
        relevant_numbers: list = Field(description="should be a list of all the relevant numbers the llm has picked up")

    parser = JsonOutputParser(pydantic_object=Numbers)

    prompt = PromptTemplate(
        input_variables = ['query'],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    ans = chain.invoke(query)
    return ans['relevant_numbers']
        

@tool
def search_tool(query : str):
    """ 
    Takes a user query and performs an online search to provide
    an answer to that query.
    """
    search = DuckDuckGoSearchRun()
    answer = search.run(query)
    return answer

@tool
def create_embeddings(query : str):
    """
    Uses the content from the file path to create embeddings on the inputted data. Using the query
    provided, perform a similarity search on the query and return the output. If you can not provide a response,
    say I don't know rather than providing an incorrect response.
    """

    #file_path = r'C:\GenAI\Final Project\GenAIGroupProject\apple.txt'
    file_path = r'./apple.txt'
    file_path = file_path.strip()
    loader = TextLoader(file_path, encoding= 'utf-8')
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunk = text_splitter.split_documents(documents=document)
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    vectorstore = FAISS.from_documents(chunk, embeddings)

    docs = vectorstore.similarity_search(query)
    return docs[0].page_content


@tool
def query_database(query : str):
    """
    This function queries the SQL database based on the query that the user inputs.
    This will provide insight into the current client(s) stock portfolio from the portfolio table in the database.
    """

    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0)
    db = SQLDatabase.from_uri('sqlite:///portfolio_allocations.db')
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    result = agent_executor.invoke(query)
    

    client = get_numbers(query)
    for i in client:
        response_schemas = [
        ResponseSchema(name=f'Client {i}', description=f"A dictionary containing the stock tickers and their percentage allocations for each client {i}.")
        ]
      
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate(
    template="create a new key, value pair for every client mentioned in the question.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    ans = chain.invoke({"question": result})

    return ans


    
    


@tool
def portfolio_allocation(query : str):
    """
    Based on the allocation stratergy provided as a query, context on the comapany's recent financials from their 10-k reports
    and their up-to-date status from an online search, change the allocation in the
    client stock portfolios in the SQL database.
    """
 
    template = """
    You are a helpful assistant who is going to allocate different weights based on context and the invest stratergy that user wants.
    In order to retreieve the context, use the search_tool and the create_embedding tool to search 'Tell me about the fininical siutuation of
    AAPL, MSFT, NVIDIA'. Combining this with the allocation stratergy that the user picks, redistribute the allocation percentages. Give exact figures 
    as to how this is to be used for each stock. You should also give a detailed explanation as to why you came up with these figures. Use the context
    to provide statistics in order to boost your reasoning. Make sure the reasoning is also returned in the final answwer.

    Query : {query}
    {answer}
    """
   
    examples = [
    {
        'query': 'Reallocte the portfolio for client 1 with a balanced strategy',
        'answer': 'The stock allocations for client_id 1 equal across all stocks, totalling 100.'
    },

    {
        'query': 'Reallocte the portfolio for client 1 with a risk-based strategy',
        'answer': """
                        Allocating more to stocks with lower perceived risk. 
                        From my 10k data and online search, Company 1 has strong financials
                        and minimal risk but Comapany 2 has more risk. As a result, the stock
                        allocation fror client_id 1 would have a higher percentage for Company 1
                        and a lower percentage for Company 2. Company 3 would make up the rest of the portfolio.
                        """
    },

    {
        'query': 'Reallocte the portfolio for client 1 with a return-based strategy',
        'answer' : """
                        Allocating more to stocks with higher expected returns.
                        From my 10k data and online search, Company 3 has higher expected growth rates and returns
                        whereas Company 2 has a declining industry position and poor management quality. As a result, the stock
                        allocation fror client_id 1 would have a high percentage for Company 3
                        and a low percentage for Company 2 and Company 1 would make up the rest of the portfolio.
                        """
    }
    ]
    c1 = create_embeddings("what is the current financial outlook of AAPL, MSFT, NVIDIA")
    c2 = search_tool("what is the current financial outlook of AAPL, MSFT, NVIDIA")

    context = c1 + c2

    example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template=template
    )

    prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Query: {input}, Context: {context}",
    input_variables=["input", "context"],
    )

    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0.2)

    chain = prompt | llm

    #print(f"context: {context}")
    ans = chain.invoke({'input' : query, 'context':context})
    
    query_database(f"set the stock allocations for the client and percentages from {ans.content} provide the sql query to do so and execute it on the database")
    
    return ans.content

    
tools = [Tool(name="search_tool", func=search_tool, description="Tool for performing online search operations"),
         Tool(name="create_embeddings", func=create_embeddings, description="Uses the content from the file path to create embeddings on the inputted data. Using the query provided, perform a similarity search on the query and return the output. If you can not provide a response,say I don't know rather than providing an incorrect response."),
         Tool(name='query_database', func=query_database, description= "This function queries the SQL database based on the query that the user provides to provide insight into the current client stock portfolio"),
         Tool(name='portfolio_allocation', func=portfolio_allocation, description='This function uses context from the vector database and online searches as well as the investment stratergy the user specifies to determine the stock allocation splits')]