#Imports
# Specifying all the imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.few_shot import FewShotPromptTemplate
import os
from generate_synthetic_data import get_share_value

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from vector_database_rawtxt import create_milvus_connection_read
from sentence_transformers import SentenceTransformer


#Tools
load_dotenv()

def get_numbers(query: str):
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0)
    template = """
    You are smart agent that can idenify all the client ids to solve the query. 
    If no start point for a client is mentioned, assume client 1 is the first client.
    id. From the query, return all the ids that the query refers to as a list of integers.
    {format_instructions}
    {query}
    """

    class Numbers(BaseModel):
        numbers: str = Field(description="should say 'the numbers extracted from the query'")
        relevant_numbers: list = Field(description="should be a list of all the relevant numbers the llm has picked up.")

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
def query_embeddings(query : str):
    """
    Loads in a vector store to give precise information about recent company financial information taken from their 
    10-k reports. 
    A similarity search is performed from your query on these reports and relevant information is returned to you.
    """

    vec_collection = create_milvus_connection_read()


    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_query_qn = model.encode(query)

    search_params = {
    "metric_type": "L2"
    }

    result = vec_collection.search(
        [embedded_query_qn.tolist()], 
        "embedding",
        search_params,
        limit=5
    )

    chunk_list = []
    res = []
    for hit in result:
        for id in hit:
            chunk_list.append(id.id)

    for id in chunk_list:
        search_result = vec_collection.query(expr=f"pk == {id}", output_fields=["*"])
        res.append(search_result[0]["plain_text"])

    #print(res)
    return res


@tool
def query_database(query : str):
    """
    This function queries the SQL database based on the query that the user inputs.
    This will provide insight into the current client(s) stock portfolio from the portfolio3 table in the database.
    """

    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0)
    db = SQLDatabase.from_uri('sqlite:///portfolio_allocations.db')
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    result = agent_executor.invoke(query)
    return result



@tool
def portfolio_allocation(query : str):
    """
    Based on the allocation stratergy provided as a query, context on the comapany's recent financials from their 10-k reports
    and their up-to-date status from an online search, change the allocation in the
    client stock portfolios in the SQL database.
    """
 
    template = """
    You are a helpful assistant who is going to allocate different weights based on context and the invest strategy that user wants.
    In order to retreieve the context, use the search_tool and the query_embedding tool to search 'Tell me about the fininical siutuation of
    AAPL, MSFT, NVIDIA'. Combining this with the allocation strategy that the user picks, redistribute the allocation percentages. Give exact figures 
    as to how this is to be used for each stock. You should also give a detailed explanation as to why you came up with these figures. Use the context
    to provide statistics in order to boost your reasoning. Make sure the reasoning is also returned in the final answwer. Use the result to update the curent
    values in the SQL database to those that you outputted.

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
    c1 = query_embeddings("what is the current financial outlook of AAPL, MSFT, NVIDIA")
    c2 = search_tool("What is the current financial outlook of AAPL, MSFT, NVIDIA")

    context = c1.append(c2)

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

    
    query_database(f"set the stock allocations for the client and percentages from {ans.content} provide the sql query to do so and execute it on the database.")

    get_share_value()
    return ans.content

tools = [
         Tool(name="query_embeddings", func=query_embeddings, description="Tool for performing vector similarity search on comapny financial data from their 10-K reports."),
         Tool(name="search_tool", func=search_tool, description="Tool for performing online search operations"),
         Tool(name='query_database', func=query_database, description= "This function queries the SQL database based on the query that the user provides to provide insight into the current client stock portfolio"),
         Tool(name='portfolio_allocation', func=portfolio_allocation, description='This function uses context from the vector database and online searches as well as the stratergy the user specifies to determine the stock allocation splits')]

def call_agent(query: str):
    template = """
    You are a helpful agent who tries to answer all questions using the query_embeddings tool FIRST.
    Answer all inputs using the query_embeddings tool FIRST and if not enough
    specific information was found from the input in the tool results, use the search_tool. If the query_embedding tool has the capability to answer the
    input question, only use this tool.

    If the terms 'client' or 'clients' are mentioned in the query, always use the query_database tool.
    If the terms 'client' or 'clients' are mentioned in the query alongside the word 'strategy', always use the portfolio_allocation tool.

    Give a detailed and clear answer.
    You have access to the following tools:
    {tools}

    Question: the input question you must answer

    Thought: you should always think about what to do

    Action: the action to take, should be one of [{tool_names}]

    Action Input: the input to the action

    Observation: the result of the action

    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer 

    Final Answer: the final answer to the ORIGINAL input question

    Begin!


    Question: {input}

    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate(input_variables=['input','tools', 'agent_scratchpad', 'tool_names'], template=template)

    # Initialize the language model
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0)


    # Create the agent using the language model and the toolset
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Execute the agent with the given input
    response = agent_executor.invoke({'input': query, 'tools' : tools, 'tool_names' : [tool.name for tool in tools]})

    return response['output'], agent_executor

#Question set:
#Embeddings
#query = "What are the strategic objectives of Apple over the next year?"
#query = "Give me the net sale value for iphones in Apple in 2022 in America"
#query =  "How much profit did Apple make in comparison to Microsoft in 2021?"

#Searches
#query = "Tell me today's Nvidia stock value"
#query = 'What is the weather in london right now?'

#SQL search:
#query = 'give me all the stock allocations for client 1 and client 2'
#query = 'give me the stock allocations for client 5'

#portfolio allocation queries 
#query = 'Reallocate the stocks for client 6 using a balanced strategy'
#query = 'Reallocate the stocks for client 5 using a risk-based strategy'
#query = 'Reallocate the stocks for client 2 using a returns-based strategy'
# resp, agent = call_agent(query=query)
# print(resp['output'])