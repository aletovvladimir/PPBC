from ..base.fedavg import FedAvg

from collections import OrderedDict

import numpy as np
import torch
from utils.model_utils import get_model

import time

import re

class PPBC(FedAvg):
    def __init__(self, theta, gamma, **method_args):
        super().__init__()

        self.norm_keywords = ['running_var']# 'in', 'instance_norm']

        self.theta = theta
        self.gamma = gamma

        # self.epoch_method = method_args.get("epoch_method", "random")
        # self.iter_method = method_args.get("iter_method", "None")
        self.epoch_k = method_args.get("epoch_k", 3)
        self.iter_k = method_args.get("iter_k", 1)
        self.iterations = method_args.get("iterations", 1)

        # self.trust_sample_amount = method_args.get("trust_sample_amount", 50)
        # self.momentum_beta = method_args.get("momentum_beta", 0.1)
        # self.q_m = method_args.get("q_m", 1.0)

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        self.num_clients = cfg.federated_params.amount_of_clients

        self.current_errors_from_clients = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }
        self.final_errors = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }

        if "pathology" in cfg.dataset.data_sources.train_directories[0]:
            self.distribution = np.load(self.cfg.dataset.distribution_info)
        else:
            self.distribution = [len(self.df) // self.num_clients] * self.num_clients

    def get_communication_content(self, rank, cur_iter, first_model):

        global_iteration = self.iterations * self.cur_round + cur_iter

        if cur_iter == 0:
            return {
                "update_model": {
                    k: v.cpu() for k, v in self.server.global_model.state_dict().items()
                },
            }

        checker = global_iteration % self.num_clients
        # In fedavg we need to send model after aggregate
        if rank == checker or rank == (checker + 1) % self.num_clients:
            return {
                "update_model": {
                    k: v.cpu() for k, v in self.server.global_model.state_dict().items()
                },
            }
        else:
            grad = self.server.client_gradients[rank]
            return {
                "update_model": {
                    k: v.cpu() + grad[k].cpu() for k, v in first_model.items()
                },
            }


    def parse_communication_content(self, client_result):
        # In fedavg we recive result_dict from every client
        self.server.set_client_result(client_result)
        print(f"Client {client_result['rank']} finished in {client_result['time']}")

        if self.cfg.federated_params.print_client_metrics:
            # client_result['client_metrics'] = (loss, metrics)
            client_loss, client_metrics = (
                client_result["client_metrics"][0],
                client_result["client_metrics"][1],
            )
            print(client_metrics)
            print(f"Validation loss: {client_loss}\n")

    def train_round(self, itn, first_model):
        for batch_idx, clients_batch in enumerate(self.clients_loader):
            print(f"Current batch of clients is {clients_batch}", flush=True)

            # Send content to clients to start local learning
            for pipe_num, rank in enumerate(clients_batch):
                # print(f'DEBUG INFO', flush = True)
                content = self.get_communication_content(rank, itn, first_model)
                self.server.send_content_to_client(pipe_num, content)

            # Waiting end of local learning and recieve content from clients
            for pipe_num, rank in enumerate(clients_batch):
                # print("START REVERSE COMM DEBUG", flush=True)
                content = self.server.rcv_content_from_client(pipe_num)
                self.parse_communication_content(content)

            # Manager reinit clients with new ranks
            self.manager.step(batch_idx)

    # =========================================================================#
    #                           Compressor Utilities                          #
    # =========================================================================#

    def deterministic_compressor(self, mode, itn=None):
        if mode == "epoch":
            self.epoch_compress_politic = self.current_politic
        elif mode == "iter":
            self.iter_compress_politic = torch.zeros_like(self.current_politic)
            self.iter_compress_politic[itn] = self.epoch_compress_politic[itn]

    def epoch_compressor(self):
        self.deterministic_compressor(mode="epoch")

    def iter_compressor(self, itn):
        checker = (self.cur_round * self.iterations + itn) % self.num_clients
        self.deterministic_compressor(mode="iter", itn=checker)
        print(f"On this iteration we choose the {itn} client and the distribution is {self.iter_compress_politic}")

    # =========================================================================#
    #                       Main algorithm functionality                      #
    # =========================================================================#

    def get_init_point(self):
        aggregated_weights = self.server.global_model.state_dict()
        for rank in range(self.num_clients):
            client_errors = self.final_errors[f"client {rank}"]

            for key, _ in aggregated_weights.items():
                aggregated_weights[key] = _ + client_errors[key] # / self.iterations
                parts = key.lower().split('.')
                if any(any(kw in part for kw in self.norm_keywords) for part in parts):
                    aggregated_weights[key] = torch.clip(aggregated_weights[key], min=1e-8)

        self.server.global_model.load_state_dict(aggregated_weights)

    def get_data_size(self):
        current_data_size = torch.sum(
            self.iter_compress_politic
            * torch.tensor(self.distribution).to(self.server.device)
            * self.num_clients
        )
        return current_data_size

    def get_errors_on_iter(self, itn):
        aggregated_weights = self.server.global_model.state_dict()

        data_size = self.get_data_size()
        print(f"now we use {data_size} objects from dataset")

        for rank in range(self.num_clients):
            client_grad = self.server.client_gradients[rank]
            current_client_error = self.current_errors_from_clients[f"client {rank}"]
            final_client_error = self.final_errors[f"client {rank}"]
            client_politic = self.iter_compress_politic[rank].to(self.server.device)

            for key, grads in client_grad.items():
                parts = key.lower().split('.')
                
                self.current_errors_from_clients[f"client {rank}"][key] = (
                    current_client_error[key]
                    + (1 - self.theta)
                    * (1 / self.num_clients - client_politic)
                    * grads.to(self.server.device)
                )

                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + self.gamma
                    * (1 - self.theta)
                    * grads.to(self.server.device)
                    * client_politic
                    + self.gamma
                    * self.theta
                    * final_client_error[key]  #/ self.iterations
                )

            if itn == self.iterations - 1:
                self.final_errors[
                    f"client {rank}"
                ] = self.current_errors_from_clients[f"client {rank}"]
                print("final errors saved!")
            if any(any(kw in part for kw in self.norm_keywords) for part in parts):
                aggregated_weights[key] = torch.clip(aggregated_weights[key], min=1e-8)
        return aggregated_weights

    def init_errors(self):
        if self.cur_round != 0:
            self.epoch_compressor()
            for rank in range(self.num_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f"client {rank}"][
                        key
                    ] = torch.zeros_like(_).to(self.server.device)
            return
        else:
            self.current_politic = (
                torch.ones(self.num_clients).to(self.server.device) / self.num_clients
            )
            self.epoch_compressor()
            for rank in range(self.num_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f"client {rank}"][
                        key
                    ] = torch.zeros_like(_).to(self.server.device)
                    self.final_errors[f"client {rank}"][key] = torch.zeros_like(_).to(
                        self.server.device
                    )
            print("creating the errors is done")

            return

    def process_clients(self):
        self.init_errors()

        self.get_init_point()
        # self.get_clients()
        aggregated_weights = 0
        first_model = {k : v.detach().cpu().clone() for k, v in self.server.global_model.state_dict().items()}

        for itn in range(self.iterations):
            self.iter_compressor(itn)

            print(f"start the {itn} iteration")
            self.train_round(itn, first_model)
            
            aggregated_weights = self.get_errors_on_iter(itn)

            self.server.global_model.load_state_dict(aggregated_weights)

            print("processing is done")

    def check_final_errors(self):
        for i in range(self.num_clients):
            c = 0
            for key, w in self.final_errors[f"client {i}"].items():
                if np.allclose(w.cpu(), torch.zeros_like(w).cpu()):
                    c += 1
            if c == len(self.final_errors[f"client {i}"].items()):
                print(f"all errors for {i} client are equals to zeros")

    def begin_train(self):
        self.create_clients()
        self.clients_loader = self.manager.batches
        self.server.global_model = get_model(self.cfg)

        for round in range(self.rounds):
            begin_round_time = time.time()
            self.cur_round = round
            print(f"\nRound number: {round} of {self.rounds}")

            _ = self.server.test_global_model()

            print("\nTraining started\n")

            self.process_clients()

            self.server.save_best_model(round)

            self.check_final_errors()

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        self.stop_train()
