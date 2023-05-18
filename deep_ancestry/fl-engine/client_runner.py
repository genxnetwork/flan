from dataclasses import dataclass
import logging
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Union
import time
from grpc import RpcError
import os
import socket

from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import ActiveRun
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy
import torch
import flwr
import torch

from fl.federation.client import FLClient, MLFlowMetricsLogger, MetricsLogger
from local.experiment import NNExperiment, TGNNExperiment
from fl.federation.callbacks import PlotLandscapeCallback, CovariateWeightsCallback


@dataclass
class NodeArgs:
    server: str
    port: str
    node_index: int
    node_name: str
    log_dir: str
    threads: int = 1
    

class Node:
    def __init__(self, args: NodeArgs):
        """Process for training on one dataset node

        Args:
            server_url (str): Full url to flower server
            log_dir (str): Logging directory, where node-{node_index}.log file will be created
            mlflow_info (MlflowInfo): Mlflow parent run and experiment IDs
            queue (Queue): Queue for communication between processes
            cfg (DictConfig): Full config with fields model, optimizer, scheduler, experiment, data
            trainer_info (TrainerInfo): Where to train node
        """
        self.args = args
        os.environ['MASTER_PORT'] = str(args.port)
        torch.set_num_threads(args.threads)


    def _configure_logging(self):
        # to disable printing GPU TPU IPU info for each trainer each FL step
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
        # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        self.logger = logging.getLogger(f'node-{self.node_index}.log')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.FileHandler(os.path.join(self.log_dir, f'node-{self.node_index}.log')))

        # logging.basicConfig(filename=os.path.join(self.log_dir, f'node-{self.node_index}.log'), level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')

    def log(self, msg):
        self.logger.info(msg)

    def _start_client_run(self, client: MlflowClient,
                        parent_run_id: str,
                        experiment_id: str,
                        tags: Dict[str, Any]) -> ActiveRun:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        # logging.info(f'starting to create mlflow run with parent {parent_run_id}')
        run = client.create_run(
            experiment_id,
            tags=tags,
        )
        # logging.info(f'run info id in _start_client_run is {run.info.run_id}')
        return mlflow.start_run(run.info.run_id, nested=True)

    def _train_model(self, client: FLClient) -> bool:
        """
        Trains a model using {client} for FL

        Args:
            client (FLClient): Federation Learning client which should implement weights exchange procedures.
        """
        for i in range(2):
            try:
                print(f'starting numpy client with server {client.server}')
                flwr.client.start_numpy_client(f'{client.server}', client)
                return True
            except RpcError as re:
                # probably server slurm job have not started yet
                print(re)
                time.sleep(20)
                continue
            except Exception as e:
                print(e)
                self.logger.error(e)
                raise e
        return False

    def create_callbacks(self):
        """Init FL client callbacks if they are specified in cfg

        Returns:
            Optional[List[ClientCallback]]: List of initialized callbacks or None
        """
        callbacks = []

        callbacks_desc = self.cfg.get('callbacks', None)
        if callbacks_desc is None:
            return callbacks
        
        return callbacks        

    def run(self) -> None:
        """Runs data loading and training of node
        """
        self._configure_logging()
        # logging.info(f'logging is configured')
        mlflow_client = MlflowClient()
        self.experiment.load_data()
        
        metrics_logger = MLFlowMetricsLogger()
        client_callbacks = self.create_callbacks()
        client = FLClient(self.server_url,
                          self.experiment,
                          self.cfg,
                          self.logger,
                          metrics_logger,
                          client_callbacks)

        self.log(f'client created, starting mlflow run for {self.node_index}')
        with self._start_client_run(
            mlflow_client,
            parent_run_id=self.mlflow_info.parent_run_id,
            experiment_id=self.mlflow_info.experiment_id,
            tags={
                'description': self.cfg.experiment.description,
                'node_index': str(self.node_index),
                'phenotype': self.cfg.data.phenotype.name,
                'split': self.cfg.split.name,
                # 'snp_count': str(self.snp_count),
                # 'sample_count': str(self.sample_count)
            }
        ):
            mlflow.log_params(OmegaConf.to_container(self.cfg.node, resolve=True))
            self.log(f'Started run for node {self.node_index}')
            
            self._train_model(client)
