"""File for running ML FLow server. ML Flow is used here for:
1. Tracking: To log and compare experiments.
2. Evaluation: To generate evaluation metrics for each model which is run.
"""

# IMPORTS --------------------------------------------------------------------------------------------------------------

import subprocess

# ML-FLOW FUNCTIONS ----------------------------------------------------------------------------------------------------

import sys
print(sys.executable)

#command = ['python', '-m', 'mlflow', 'server', '--host', '127.0.0.1', '--port', '8080']
command = ['python', '-m', 'pip', 'list']

try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Output:", result.stdout)

except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
    print("Return Code:", e.returncode)
