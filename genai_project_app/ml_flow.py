"""File for running ML FLow server. ML Flow is used here for:
1. Tracking: To log and compare experiments.
2. Evaluation: To generate evaluation metrics for each model which is run.
"""

# IMPORTS & SETUP ------------------------------------------------------------------------------------------------------

from typing import Union, List

import time
import mlflow
import atexit
import logging
import textstat
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain import hub
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from mlflow.tracking import MlflowClient
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableSequence
from mlflow.models.evaluation.base import EvaluationResult
from langchain.agents import AgentExecutor, create_react_agent
from mlflow.metrics.genai import answer_similarity, faithfulness, answer_correctness, answer_relevance

logging.basicConfig(level=logging.INFO)

# ML-FLOW FUNCTIONS ----------------------------------------------------------------------------------------------------


def mlflow_server(port: int = 8080) -> subprocess.Popen:
    """Uses command line code to start an ML-Flow server at the port.

    Args:
        port (int, optional): The port at which to run the server. Defaults to 8080.
    
    Returns:
        server_process (subprocess.Popen): The Popen object representing the server. 
    """
    server_process = None
    command = ['python', '-m', 'mlflow', 'server', '--host', '127.0.0.1', '--port', str(port)]

    try:
        server_process = subprocess.Popen(command)
        logging.info('Successfully running ML-Flow server. The server will terminate at the end of runtime.')
        atexit.register(_close_mlflow_server, server_process)

    except subprocess.CalledProcessError as e:
        logging.warning('Unable to run ML-Flow server.')
        print("Error:", e.stderr)
        print("Return Code:", e.returncode)

    return server_process


def _close_mlflow_server(server_process: subprocess.Popen) -> None:
    """Closes the supplied server process.

    Args:
        server_process (subprocess.Popen): The server object in question.
    """
    server_process.terminate()
    server_process.wait()
    server_process.kill()
    server_process.wait()
    logging.info("MLFlow server terminated.")


def get_info_on_runs(experiment_name: str = 'mlflow_development', tracking_uri: str = "http://localhost:8080/") -> str:

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return f"Experiment '{experiment_name}' not found."

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string="",
                              run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)

    response = []

    for run in runs:
        response.append("-" * 120)
        response.append(f"Run ID: {run.info.run_id}")

        extra_info = {'Parameters': run.data.params,
                      'Metrics': run.data.metrics,
                      'Tags': run.data.tags,
                      'Artifacts': client.list_artifacts(run.info.run_id)}

        for info_name, info_data in extra_info.items():
            if len(info_data) != 0:
                response.append(f"{info_name}: {info_data}")

    return '\n'.join(response)


def delete_all_runs(experiment_name: str = 'mlflow_development', tracking_uri: str = 'http://localhost:8080/') -> None:
    """Deletes the runs of the specified experiment if it exists.

    Args:
        tracking_uri (_type_, optional): The URI at which the experiment should be sought. Defaults to
                                         'http://localhost:8080/'.
        experiment_name (str, optional): The name of the experiment to be deleted. Defaults to 'mlflow_development'.

    Raises:
        ValueError: Raised if the experiment does not exist.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    client.delete_experiment(experiment.experiment_id)


# AGENT FUNCTIONS ------------------------------------------------------------------------------------------------------


def create_agent(model_name: str = 'gpt-3.5-turbo-instruct') -> AgentExecutor:
    """Creates an agent model based on the react prompt.

    Args:
        model_name (str, optional): The name of the model from which the agent is derived. Defaults to
                                    'gpt-3.5-turbo-instruct'.

    Returns:
        AgentExecutor: The agent instance.
    """
    prompt = hub.pull('hwchase17/react')
    llm = OpenAI(model_name=model_name, temperature=0.2)

    agent = create_react_agent(llm, tools=[], prompt=prompt)

    return AgentExecutor(agent=agent, tools=[], handle_parsing_errors=True, verbose=False, max_iterations=5,
                         return_intermediate_steps=True)


def get_agent_thoughts(agent_result: dict) -> str:
    """Inspects the output of an agent model and pulls out the thoughts which lead to the conclusion. 

    Args:
        agent_result (dict): Dict representing results of a query to an agent model.

    Returns:
        str: A string representing the thoughts of the agent model.
    """
    thoughts = []
    print(agent_result)
    for step in agent_result['intermediate_steps']:
        thoughts += [f'Tool: {step[0].tool}']
        thoughts += [f'Log: {step[0].log}']
        thoughts += '\n'

    return '\n'.join(thoughts)


# LLM FUNCTIONS --------------------------------------------------------------------------------------------------------


def create_example_llm() -> RunnableSequence:
    """Creates a simple chat LLM which models an investment manager. The LLM only returns strings.

    Returns:
        RunnableSequence: A LangChain sequence representing the LLM.
    """
    example_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0)

    example_prompt = PromptTemplate(input_variables=['inputs', 'context'],
                                    template=("You're a investment manager. Using the context provided, "
                                              + "reply to the question below to the best of your ability:\n"
                                              + "Question:\n{inputs}\nContext:\n{context}"))

    def _get_content(model_return):
        return model_return.content

    get_content_lambda = RunnableLambda(_get_content)

    example_model = example_prompt | example_llm | get_content_lambda

    return example_model


# EVALUATION -----------------------------------------------------------------------------------------------------------


def success_rate(question: str, response: str) -> Union[int, None]:
    """Metric for whether a question was successfully answered, determined using LLM as a judge.

    Args:
        question (str): The question asked of a LLM.
        response (str): The response given by an LLM.

    Returns:
        Union[int, None]: 1 if correctly answered, 0 if incorrectly answered, None if indeterminate.
    """
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.2)
    success = llm.invoke('Determine if the following question was successfully answered. If it was, return ONLY the '
                         + f'number 1, else ONLY return the number 0.\nQuestion: {question}\nAnswer: {response}')

    try:
        return int(success)
    except Exception:
        return None


def evaluate_agent(agent: AgentExecutor, questions: List[str], experiment_name: str = 'mlflow_development') -> None:
    """Evaluates the given agent with the given questions, for a set of metrics:
    1. ARI Score.
    2. Success Rate.
    3. Response Time.

    Args:
        agent (AgentExecutor): The agent with which to response to queries.
        questions (List[str]): A list of questions to ask the model.
        experiment_name (str, optional): The name of the ML Flow experiment. Defaults to 'mlflow_development'.
    """
    all_score_values = {}

    for question in tqdm(questions, desc='Evaluating agent on questions...'):
        start_time = time.time()
        result = agent.invoke({"input": question})
        end_time = time.time()

        thoughts = get_agent_thoughts(result)

        if 'ari_score' not in all_score_values.keys():
            all_score_values['ari_score'] = []
        all_score_values['ari_score'].append(textstat.automated_readability_index(result['output']))

        if 'success_rate' not in all_score_values.keys():
            all_score_values['success_rate'] = []
        all_score_values['success_rate'].append(success_rate(result['input'], result['output']))

        if 'response_time' not in all_score_values.keys():
            all_score_values['response_time'] = []
        all_score_values['response_time'].append(end_time - start_time)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        for score_name, score_values in all_score_values.items():
            mlflow.log_metric(score_name + '_mean', round(np.nanmean(score_values), 2))
            mlflow.log_metric(score_name + '_variance', round(np.nanvar(score_values), 2))


def evaluate_llm(llm_to_evaluate: RunnableSequence, evaluation_dataset: pd.DataFrame,
                 judge_model: str = "openai:/gpt-3.5-turbo", experiment_name: str = "mlflow_development"
                 ) -> EvaluationResult:
    """Evaluates the provided LLM using the provided dataset on a set of metrics:
        faithfulness, answer_similarity, answer_correctness, answer_relevance.

    Args:
        llm_to_evaluate (RunnableSequence): The LLM to evaluate.
        evaluation_dataset (pd.DataFrame): The dataset on which the evaluation is to be performed. Must contain the
                                           columns: 'inputs', 'context', and 'targets'.
        judge_model (str, optional): The model with which the judging should be performed. Defaults to
                                     "openai:/gpt-3.5-turbo".
        experiment_name (str, optional): The name of the experiment in ML-Flow. Defaults to "mlflow_development".

    Returns:
        EvaluationResult: The result of the model evaluation run.
    """
    faithfulness_metric = faithfulness(model=judge_model)
    answer_relevance_metric = answer_relevance(model=judge_model)
    answer_similarity_metric = answer_similarity(model=judge_model)
    answer_correctness_metric = answer_correctness(model=judge_model)
    
    extra_metrics = [faithfulness_metric, answer_similarity_metric, answer_correctness_metric, answer_relevance_metric]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run: 
        _logged_model = mlflow.langchain.log_model(llm_to_evaluate, artifact_path="model")

        mlflow.log_param("model", llm_to_evaluate)
        results = mlflow.evaluate(_logged_model.model_uri, evaluation_dataset, model_type="question-answering",
                                  targets="targets", extra_metrics=extra_metrics,
                                  evaluator_config={'col_mapping': {"inputs": "predictions"}})

        mlflow.log_metrics(results.metrics)

    return results
