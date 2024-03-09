# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import os
import time

from azureml.core import Run
from pathlib import Path

import pickle
import mlflow
import mlflow.sklearn


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--best_model', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")   

    # parse args
    args = parser.parse_args()
    print("Path: " + args.best_model)
    # return args
    return args


def main(args):
    """
    Register Model Example
    """

    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    print("Deploy flag: ", deploy_flag)
    
    mlflow.log_metric("deploy flag", int(deploy_flag))
    deploy_flag=1
    if deploy_flag==1:

        print("Registering ", args.model_name) 

        # load model
        model =  mlflow.sklearn.load_model(args.best_model)
        
        # log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name)               

        # Set Tracking URI
        current_experiment = Run.get_context().experiment
        tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
        print("tracking_uri: {0}".format(tracking_uri))
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(current_experiment.name)

        # Get Run ID from model path
        print("Getting model path")
        mlmodel_path = os.path.join(args.best_model, "MLmodel")
        runid = ""
        with open(mlmodel_path, "r") as modelfile:
            for line in modelfile:
                if "run_id" in line:
                    runid = line.split(":")[1].strip()

        # Construct Model URI from run ID extract previously
        model_uri = "runs:/{}/outputs/mlflow-model/".format(runid)
        print("Model URI: " + model_uri)                

        # Register the model with Model URI and Name of choice
        registered_name = args.model_name
        print(f"Registering model as {registered_name}")
        mlflow_model = mlflow.register_model(model_uri, registered_name)
        model_version = mlflow_model.version

        # write model info
        print("Writing JSON")
        dict = {"id": "{0}:{1}".format(args.model_name, model_version)}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)        

    else:
        print("Model will not be registered!")        


# run script
if __name__ == "__main__":

    mlflow.start_run()

    # parse args
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.best_model}",
        f"Evaluation output path: {args.evaluation_output}",        
    ]

    for line in lines:
        print(line)    

    # run main function
    main(args)

    mlflow.end_run()      


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers trained ML model if deploy flag is True.
"""
"""
import argparse
from pathlib import Path
import pickle
import mlflow

import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main(args):
    '''Loads model, registers it if deply flag is True'''

    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    mlflow.log_metric("deploy flag", int(deploy_flag))
    deploy_flag=1
    if deploy_flag==1:

        print("Registering ", args.model_name)

        # load model
        model =  mlflow.sklearn.load_model(args.model_path) 

        # log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name)

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version

        # write model info
        print("Writing JSON")
        dict = {"id": "{0}:{1}".format(args.model_name, model_version)}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)

    else:
        print("Model will not be registered!")

if __name__ == "__main__":

    mlflow.start_run()
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    """
