"""File for running ML FLow server. ML Flow is used here for:
1. Tracking: To log and compare experiments.
2. Evaluation: To generate evaluation metrics for each model which is run.
"""

# IMPORTS & SETUP ------------------------------------------------------------------------------------------------------

import atexit
import logging
import subprocess

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
