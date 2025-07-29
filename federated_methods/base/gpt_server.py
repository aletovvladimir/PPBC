import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from hydra.utils import instantiate

from utils.utils import create_model_info
from utils.losses import get_loss
from utils.data_utils import read_dataframe_from_cfg, get_dataset_loader
from utils.metrics_utils import (
    calculate_metrics,
    stopping_criterion,
    check_metrics_names,
)


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client_gradients = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.server_metrics = [
            pd.DataFrame() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.test_df = read_dataframe_from_cfg(cfg, "test_directories", "server_test")
        self.test_loader = get_dataset_loader(
            self.test_df, cfg, drop_last=False, mode="test"
        )
        self.device = (
            "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )
        print(f"server has device {self.device}", flush=True)
        self.model_path = self.create_model_path()
        self.best_metrics = {
            metric: 1000 * (metric == "loss")
            for metric in cfg.federated_params.server_saving_metrics
        }
        check_metrics_names(self.best_metrics)
        self.metric_aggregation = cfg.federated_params.server_saving_agg
        assert self.metric_aggregation in [
            "uniform",
            "weighted",
        ], f"federated_params.server_saving_agg can be only ['uniform', 'weighted'], you provide: {self.best_metrics}"
        self.best_round = 0
        self.last_test_metrics = (None, None)
        self.last_trust_metrics = (None, None)

    def eval_fn(self, dataset):
        self.global_model.to(self.device)
        self.global_model.eval()
        
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for _, batch in enumerate(getattr(self, f"{dataset}_loader")):
                _, (input_ids, target_ids) = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                _, loss = self.global_model(input_ids, targets=target_ids)

                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)

        avg_loss = total_loss / total_tokens
        setattr(self, f"{dataset}_loss", avg_loss)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity

    def test_global_model(self, dataset="test", require_metrics=True):
        avg_loss, perplexity = self.eval_fn(dataset)

        if require_metrics:
            print(f"\n[Server] {dataset.capitalize()} Results:")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Perplexity: {perplexity:.4f}")

        setattr(self, f"last_{dataset}_metrics", ({"perplexity": perplexity}, None))

    def set_client_result(self, client_result):
        # Put client information in accordance with his rank
        self.client_gradients[client_result["rank"]] = client_result["grad"]
        self.server_metrics[client_result["rank"]] = client_result["server_metrics"]

    def save_best_model(self, round):
        # Collect metrics from clients
        # server_metrics = (metrics, val_loss, len(val_df))
        server_metrics = [metrics[0] for metrics in self.server_metrics]

        val_losses = [metrics[1] for metrics in self.server_metrics]

        val_len_dfs = [metrics[2] for metrics in self.server_metrics]

        weights = [val_len_df / sum(val_len_dfs) for val_len_df in val_len_dfs]

        metrics_names = server_metrics[0].index

        if self.metric_aggregation == "uniform":
            # Uniform metrics agregation
            val_loss = np.mean(val_losses)
            metrics = pd.concat(server_metrics).groupby(level=0).mean()

        if self.metric_aggregation == "weighted":
            # Weighted metrics aggregation
            val_loss = np.sum(
                [loss * weight for loss, weight in zip(val_losses, weights)]
            )
            metrics = sum(
                weight * metric for weight, metric in zip(weights, server_metrics)
            )

        metrics = metrics.reindex(metrics_names)
        print(f"\nServer Valid Results:\n{metrics}")
        print(f"Server Valid Loss: {val_loss}")
        # Update best metrics
        epochs_no_improve, best_metrics = stopping_criterion(
            val_loss, metrics, self.best_metrics, epochs_no_improve=0
        )
        if epochs_no_improve == 0 and val_loss is not np.nan:
            print("\nServer model saved!")

            prev_model_path = f"{self.model_path}_round_{self.best_round}.pt"
            if os.path.exists(prev_model_path):
                os.remove(prev_model_path)

            self.best_metrics = best_metrics
            self.best_round = round

            checkpoint_path = f"{self.model_path}_round_{self.best_round}.pt"

            model_info = create_model_info(
                model_state=self.global_model.state_dict(),
                metrics=self.last_test_metrics,
                checkpoint_path=checkpoint_path,
                cfg=self.cfg,
            )
            torch.save(model_info, checkpoint_path)
        # Print comparing results
        metrics.loc["loss"] = val_loss
        print(f"\nCriterion metrics:")
        for k, v in self.best_metrics.items():
            print(
                f"Current {k}: {metrics.loc[k].mean()}\nBest {k}: {v}\nBest round: {self.best_round}\n",
            )
        return metrics

    def create_model_path(self):
        self.target_label_names = [self.cfg.dataset.data_name]

        return f"{self.cfg.single_run_dir}/{type(instantiate(self.cfg.federated_method, _recursive_=False)).__name__}_{'_'.join(self.target_label_names)}"

    def send_content_to_client(self, pipe_num, content):
        # Send content to client
        self.pipes[pipe_num].send(content)

    def rcv_content_from_client(self, pipe_num):
        # Get content from current client
        client_content = self.pipes[pipe_num].recv()

        return client_content

    def shutdown_client(self, rank):
        # Send shutdown message
        self.pipes[rank].send({"shutdown": None})

    def reinit_client(self, rank, new_rank, client_cls, pipe):
        self.pipes[rank].send({"reinit": [new_rank, client_cls, pipe]})
