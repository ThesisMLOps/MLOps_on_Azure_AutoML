# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities._job.automl.tabular import TabularFeaturizationSettings
from azure.ai.ml.automl import classification

import json
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser("Deploy Training Pipeline")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--data_name", type=str, help="Data Asset Name")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--enable_monitoring", type=str, help="Enable Monitoring", default="false")
    parser.add_argument("--table_name", type=str, help="ADX Monitoring Table Name", default="taximonitoring")
    
    args = parser.parse_args()

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    
    credential = DefaultAzureCredential()
    try:
        ml_client = MLClient.from_config(credential, path='config.json')

    except Exception as ex:
        print("HERE IN THE EXCEPTION BLOCK")
        print(ex)

    try:
        print(ml_client.compute.get(args.compute_name))
    except:
        print("No compute found")



    print(os.getcwd())
    print('current', os.listdir())

    # Create pipeline job

    # 1. Define components
    parent_dir = "data-science/src"
    
    prep_data = command( 
        name="prep_data",
        display_name="prep-data",
        code=os.path.join(parent_dir, "prep"),
        command="python prep.py \
                --raw_data ${{inputs.raw_data}} \
                --train_data ${{outputs.train_data}}  \
                --test_data ${{outputs.test_data}} \
                --enable_monitoring ${{inputs.enable_monitoring}} \
                --table_name ${{inputs.table_name}}",
        environment=args.environment_name+"@latest",
        inputs={
            "raw_data": Input(type="uri_file"),
            "enable_monitoring": Input(type="string"),
            "table_name": Input(type="string"),
            },
        outputs={
            "train_data": Output(type="mltable"),
            "test_data": Output(type="mltable"),
            }
    )

    evaluate_model = command(
        name="evaluate_model",
        display_name="evaluate-model",
        code=os.path.join(parent_dir, "evaluate"),
        command="python evaluate.py \
                --model_name ${{inputs.model_name}} \
                --best_model ${{inputs.best_model}} \
                --test_data ${{inputs.test_data}} \
                --evaluation_output ${{outputs.evaluation_output}}",
        environment=args.environment_name+"@latest",
        inputs={
            "model_name": Input(type="string"),
            "best_model": Input(type="mlflow_model"),
            "test_data": Input(type="mltable")
            },
        outputs={
            "evaluation_output": Output(type="uri_folder")
            }
    )

    register_model = command(
        name="register_model",
        display_name="register-model",
        code=os.path.join(parent_dir, "register"),
        command="python register.py \
                --model_name ${{inputs.model_name}} \
                --best_model ${{inputs.best_model}} \
                --evaluation_output ${{inputs.evaluation_output}} \
                --model_info_output_path ${{outputs.model_info_output_path}}",
        environment=args.environment_name+"@latest",
        inputs={
            "model_name": Input(type="string"),
            "best_model": Input(type="mlflow_model"),
            "evaluation_output": Input(type="uri_folder")
            },
        outputs={
            "model_info_output_path": Output(type="uri_folder")
            }
    )

    # 2. Construct pipeline
    @pipeline(
        description="AutoML HMDA Data Classification Pipeline",        
    )
    def hmda_training_pipeline(raw_data, enable_monitoring, table_name):
        
        preprocess_node = prep_data(
            raw_data=raw_data,
            enable_monitoring=enable_monitoring, 
            table_name=table_name
        )

        classification_node = classification(
            training_data=preprocess_node.outputs.train_data,
            target_column_name="action_taken",
            primary_metric="accuracy",
            enable_model_explainability=True,
            validation_data_size=0.1,
            outputs={"best_model": Output(type="mlflow_model")},
        )
        # set limits and training
        classification_node.set_limits(
            enable_early_termination=True,
            max_concurrent_trials=1,
            max_trials=1,                                   
            timeout_minutes=30, 
            trial_timeout_minutes=10            
        )

        classification_node.set_training(
            enable_stack_ensemble=False,
            enable_vote_ensemble=False
        )                

        evaluation_node = evaluate_model(
            model_name="hmda-model",
            best_model=classification_node.outputs.best_model,
            test_data=preprocess_node.outputs.test_data
        )

        register_node = register_model(
            model_name="hmda-model",
            best_model=classification_node.outputs.best_model,
            evaluation_output=evaluation_node.outputs.evaluation_output
        )

        return {
            "pipeline_job_train_data": preprocess_node.outputs.train_data,
            "pipeline_job_test_data": preprocess_node.outputs.test_data,
            "pipeline_job_trained_model": classification_node.outputs.best_model,
            "pipeline_job_score_report": evaluation_node.outputs.evaluation_output,
        }


    pipeline_job = hmda_training_pipeline(
        Input(path=args.data_name + "@latest", type="uri_file"), args.enable_monitoring, args.table_name
    )

    # set pipeline level compute
    pipeline_job.settings.default_compute = args.compute_name
    # set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )

    ml_client.jobs.stream(pipeline_job.name)

    
if __name__ == "__main__":
    main()

#featurization=TabularFeaturizationSettings(mode="Off"),
