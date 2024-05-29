"""File for running ML FLow server. ML Flow is used here for:
1. Tracking: To log and compare experiments.
2. Evaluation: To generate evaluation metrics for each model which is run.
"""

# IMPORTS & SETUP ------------------------------------------------------------------------------------------------------

import mlflow
import atexit
import logging
import subprocess
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableSequence
from mlflow.models.evaluation.base import EvaluationResult
from mlflow.metrics.genai import answer_similarity, faithfulness, answer_correctness, answer_relevance

logging.basicConfig(level=logging.INFO)

# ML-FLOW FUNCTIONS ----------------------------------------------------------------------------------------------------


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
