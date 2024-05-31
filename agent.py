# Specifying all the imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.few_shot import FewShotPromptTemplate
import os
from sentence_transformers import SentenceTransformer


#create tools
load_dotenv()

def get_numbers(query: str):
    """Use an LLM to parse what clients the SQL query is to be run for."""

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
def query_embeddings(query : str):
    """
    Loads in a vector store to give precise information about recent company financial information taken from their 
    10-k reports. 
    A similarity search is performed from your query on these reports and relevant information is returned to you.
    """
    from vector_database_rawtxt import create_milvus_db
    vector_store = create_milvus_db()
    vector_store.load()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_query_qn = model.encode(query)

    search_params = {
    "metric_type": "L2"
    }

    result = vector_store.search(
        [embedded_query_qn.tolist()], # a single query must still be enclosed in a list
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
        search_result = vector_store.query(expr=f"pk == {id}", output_fields=["*"])
        res.append(search_result[0]["plain_text"])
    return res


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
    In order to retreieve the context, use the search_tool and the query_embedding tool to search 'Tell me about the fininical siutuation of
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
    c1 = query_embeddings("what is the current financial outlook of AAPL, MSFT, NVIDIA")
    c2 = search_tool("what is the current financial outlook of AAPL, MSFT, NVIDIA")

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

    ans = chain.invoke({'input' : query, 'context':context})
    
    query_database(f"set the stock allocations for the client and percentages from {ans.content} provide the sql query to do so and execute it on the database")
    
    return ans

    
tools = [
         Tool(name="query_embeddings", func=query_embeddings, description="Tool for performing vector similarity search on comapny financial data from their 10-K reports."),
         Tool(name="search_tool", func=search_tool, description="Tool for performing online search operations"),
         Tool(name='query_database', func=query_database, description= "This function queries the SQL database based on the query that the user provides to provide insight into the current client stock portfolio"),
         Tool(name='portfolio_allocation', func=portfolio_allocation, description='This function uses context from the vector database and online searches as well as the investment stratergy the user specifies to determine the stock allocation splits')]


def call_agent(query : str):
    #queries and agent
    template = """

    Only use the portfolio_allocation tool if the terms 'strategy' AND 'client' are provided in the query.
    Otherwise any queries involving 'client' you should only use the query_database tool. 

    If the user query requests information which doesn't require the need for a SQL database, use the query_emebddings tool.
    Only use the search_tool tool if there query involves things the query_embeddings tools does not provide an output. 

    You have access to the following tools:
    {tools}

    Question: the input question you must answer

    Thought: you should always think about what to do

    Action: the action to take, should be one of [{tool_names}]

    Action Input: the input to the action

    Observation: the result of the action

    ... (this Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer

    If your thought before the final answer includes the output from the portfolio_allocation tool, give the response by including ACTUAL percentages totalling to 100. 

    Final Answer: the final answer to the ORIGINAL input question

    Begin!


    Question: {input}

    Thought:{agent_scratchpad}
    """


    # Define the query and create the prompt
    #query = 'return the values for client_id 1 in the sql database and also tell me the current apple stock price'

    #sql queries
    #query = 'Return me the stock allocation for client 5'
    #query = 'Return me the stock allocation for client 8'
    #query = 'Return me the stock allocation for every client'
    #query = 'give me all the stock allocations from all clients up to client 3'
    #query = 'give me all the stock allocations from all clients up to client 3'
    #query = 'give me all the stock allocations for client 8'
    #query = "The Company continues to develop new technologies to enhance existing products and services and to expand the range of its offerings through research and development RD licensing of intellectual property and acquisition of thirdparty businesses and technology"

    #portfolio allocation queries 
    #query = "How much Profit did company Apple make in comparison to Microsoft in 2021?"
    #query =  "add a new client to the database with random stock allocations, provide the sql query to do so and execute it"
    #query =  "set the stock allocations for client_id 1 to 5,5,90. provide the sql query to do so and execute it on the database" #remeber to pull numbers from another query into this framework
    #query = 'Reallocate the stocks for client 1 using a risk-based stratergy'

    #search queries
    #query = 'Tell me the todays MSFT stock price'

    #embedding queries
    #query = "What are the strategic objectives of Apple over in 2023?"
    #query = "Give me a sentence from the apple 10-k report"
    #query = "Give me the net sale value for iphones in 2022"
    #query = "Give me the net sales for Nvidia in Q4 2021"
    #query = "What is the general sentiment about apple based on the 2022 report?"


    prompt = PromptTemplate(input_variables=['input','tools', 'agent_scratchpad', 'tool_names'], template=template)

    # Initialize the language model
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo', temperature=0.2)


    # Create the agent using the language model and the toolset
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Execute the agent with the given input
    response = agent_executor.invoke({'input': query, 'tools' : tools, 'tool_names' : [tool.name for tool in tools]})

    return response['output'], agent_executor

resp, agent_executor = call_agent("Give me the net sales for Nvidia in Q4 2021")

print(resp)
# print(agent_executor)