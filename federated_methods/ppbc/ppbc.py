from ..ts_momentum.ts_momentum_server import TSMomentumServer
from ..base.fedavg import FedAvg

from collections import OrderedDict

import numpy as np
import random
import torch
from utils.model_utils import get_model
from utils.utils import softmax

import time

from utils.data_utils import read_dataframe_from_cfg, get_stratified_subsample
from .ppbc_client import ScaffoldClient
import copy


class PPBC(FedAvg):
    def __init__(self, theta, gamma, **method_args):
        super().__init__()

        self.theta = theta
        self.gamma = gamma

        self.epoch_method = method_args.get("epoch_method", "random")
        self.iter_method = method_args.get("iter_method", "None")
        self.epoch_k = method_args.get("epoch_k", 3)
        self.iter_k = method_args.get("iter_k", 1)
        self.iterations = method_args.get("iterations", 1)
        self.need_errors = method_args.get("need_errors", "True")

        self.trust_sample_amount = method_args.get("trust_sample_amount", 50)
        self.momentum_beta = method_args.get("momentum_beta", 0.1)
        self.q_m = method_args.get("q_m", 1.0)

        self.method = method_args.get("method", "ppbc")
        if not ("ppbc" in self.method):
            print(
                f"for {self.method} we do not need errors, so theta and need errors params are equals to 0.0 and False"
            )
            self.theta = 0.0
            self.need_errors = False

        # Scaffold method
        if self.method == "scaffold":
            print('using scaffold method')
            self.global_lr = 1e-4 # method_args.get("global_lr", 3e-4)
            self.local_lr = 3e-4 # method_args.get("local_lr", 3e-4)
        
        self.current_iter = 0
        print(f"{self.method} algorithm, theta={self.theta}")

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        print(f"grads sent per round: {self.iter_k * self.iterations + self.num_clients * ('ppbc' in self.method)}")

        self.current_errors_from_clients = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }
        self.current_grad_approx = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }
        self.final_errors = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }

        if "pathology" in cfg.dataset.data_sources.train_directories[0]:
            self.distribution = np.load(self.cfg.dataset.distribution_info)
        else:
            self.distribution = [len(self.df) // self.num_clients] * self.num_clients

    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        _, trust_df = get_stratified_subsample(
            df=trust_df,
            num_samples=self.trust_sample_amount,
            random_state=cfg.random_state,
        )

        self.server = TSMomentumServer(cfg, trust_df)

        self.num_clients = cfg.federated_params.amount_of_clients
        self.epoch_prev_trust_scores = [1 / self.num_clients] * self.num_clients
        self.iter_prev_trust_scores = [1 / self.num_clients] * self.num_clients

    # =========================================================================#
    #                           SCAFFOLD Utilities                            #
    # =========================================================================#

    def get_scaffold_aggregation(self):
        client_gradients = self.server.client_gradients
        aggregated_weights = self.server.global_model.state_dict()
        sum_grad = OrderedDict()
        for key, value in client_gradients[0].items():
            sum_grad[key] = torch.zeros(value.shape, device = self.server.device)
        for idx, gradient in enumerate(client_gradients):
            client_politic = self.iter_compress_politic[idx]
            for key, value in self.server.global_model.named_parameters():
                if not self.server.error_feedback:
                    sum_grad[key] += client_politic * gradient[key].to(self.server.device)
                else:
                    sum_grad[key] += client_politic * (self.global_control[key].to(self.server.device) + self.clients_delta_control[idx][key])

        coef = self.global_lr * (1 / sum(self.iter_compress_politic))
        for key, _ in aggregated_weights.items():
            aggregated_weights[key] = aggregated_weights[key] + coef * sum_grad[key]
        self._update_global_control()
        return aggregated_weights

    def _init_client_cls(self):
        if self.method == "scaffold":
            self.client_cls = ScaffoldClient
            self.client_args = [self.cfg, self.df, self.local_lr]
            self.client_kwargs = {
                "client_cls": self.client_cls,
                "pipe": None,
                "rank": None,
            }
        else:
            super()._init_client_cls()

    def _init_controls(self):
        self.global_control = {}

        with torch.no_grad():
            for k, v in self.server.global_model.named_parameters():
                self.global_control[k] = torch.randn_like(v) * 1e-5

        self.clients_control = [
            copy.deepcopy(self.global_control) for _ in range(self.num_clients)
        ]

        self.clients_delta_control = [
            copy.deepcopy(self.global_control) for _ in range(self.num_clients)
        ]

    """def get_communication_content(self, rank):
        # In scaffold we need additionaly send global controls to clients
        # and their own local controls

        content = super().get_communication_content(rank)
        
        return content"""

    def parse_communication_content(self, client_result):
        # In scaffold we recive result_dict from every client
        # and additionaly receive delta_control (c_plus - local_c)

        super().parse_communication_content(client_result)
        if self.method == "scaffold":
            self.clients_delta_control[client_result["rank"]] = client_result[
                "delta_control"
            ]
            self.clients_control[client_result["rank"]] = client_result[
                "client_control"
            ]

    def _update_global_control(self):

        total_delta = OrderedDict()
        for key, value in self.global_control.items():
            total_delta[key] = torch.zeros(value.shape, device = "cpu")
        for idx, gradient in enumerate(self.clients_delta_control):
            client_politic = self.iter_compress_politic[idx].to("cpu")
            for key, value in gradient.items():
                total_delta[key] += client_politic * value.to("cpu")
                if (client_politic > 0):
                    self.clients_control[idx][key] += self.clients_delta_control[idx][key].to("cpu")

        coef = self.global_lr * (1 / sum(self.iter_compress_politic))
        for key, _ in self.global_control.items():
            self.global_control[key] = self.global_control[key] + coef.to("cpu") * total_delta[key]

        self.clients_delta_control = [
            OrderedDict(
                (k, torch.zeros_like(v)) for k, v in self.global_control.items()
            )
            for _ in range(self.num_clients)
        ]

    # =========================================================================#
    #                    Trust Score Calculation Utilities                    #
    # =========================================================================#

    def get_scores_from_gradients(self):
        prev_trust_scores = [0] * self.num_clients
        for rank in range(self.num_clients):
            cur_grads = self.server.client_gradients[rank]
            cur_flat_grad = torch.cat([grad.flatten() for grad in cur_grads.values()])

            prev_trust_scores[rank] = torch.norm(cur_flat_grad)
        return prev_trust_scores

    def get_scores_from_bant(self, server_loss, client_losses, mode="epoch"):
        uncutted_trust_scores = [
            server_loss - client_loss for client_loss in client_losses
        ]
        trust_scores = [
            max(uncutted_trust_score, 0)
            for uncutted_trust_score in uncutted_trust_scores
        ]
        if mode == "epoch":
            for i in range(self.num_clients):
                if sum(trust_scores):
                    momentum_trust_score = (
                        1 - self.momentum_beta
                    ) * self.epoch_prev_trust_scores[i] + self.momentum_beta * (
                        trust_scores[i] / sum(trust_scores)
                    )
                else:
                    momentum_trust_score = (
                        1 - self.momentum_beta
                    ) * self.epoch_prev_trust_scores[i]
                self.epoch_prev_trust_scores[i] = momentum_trust_score
        elif mode == "iter":
            for i in range(self.num_clients):
                if sum(trust_scores):
                    momentum_trust_score = (
                        1 - self.momentum_beta
                    ) * self.iter_prev_trust_scores[i] + self.momentum_beta * (
                        trust_scores[i] / sum(trust_scores)
                    )
                else:
                    momentum_trust_score = (
                        1 - self.momentum_beta
                    ) * self.iter_prev_trust_scores[i]
                self.iter_prev_trust_scores[i] = momentum_trust_score


    def get_scores_from_losses(self):
        prev_trust_scores = [0] * self.num_clients
        client_results = self.server.server_metrics
        prev_trust_scores = [metrics[1] for metrics in client_results]

        return prev_trust_scores

    def get_score_from_anglse(self):
        prev_trust_scores = [0] * self.num_clients
        avg_grad = self.get_avg_grad()
        for i in range(len(self.server.client_gradients)):
            client_grad = self.server.client_gradients[i]
            sc_prod = torch.zeros(len(client_grad))
            idx = 0
            for key, value in avg_grad.items():
                sc_prod[idx] = self.get_scalar_prod(value, client_grad[key])
                idx += 1
            prev_trust_scores[i] = torch.mean(sc_prod)
        return prev_trust_scores

    # =========================================================================#
    #               Trust Score Dispatch: Epoch vs. Iteration                 #
    # =========================================================================#

    def _epoch_count_trust_score(self):
        if "bant" in self.epoch_method:
            server_loss, client_losses = self.server.get_trust_losses()
            self.get_scores_from_bant(server_loss, client_losses, mode="epoch")

        elif "gradient_norm" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_scores_from_gradients()

        elif "loss" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_scores_from_losses()

        elif "angle" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_score_from_anglse()

        else:
            print(f"{self.epoch_method} method does not requires trust scores")

    def _iter_count_trust_score(self):
        if "bant" in self.iter_method:
            server_loss, client_losses = self.server.get_trust_losses()
            self.get_scores_from_bant(server_loss, client_losses, mode="iter")

        elif "gradient_norm" in self.iter_method:
            self.iter_prev_trust_scores = self.get_scores_from_gradients()

        elif "loss" in self.iter_method:
            self.iter_prev_trust_scores = self.get_scores_from_losses()

        elif "angle" in self.iter_method:
            self.iter_prev_trust_scores = self.get_score_from_anglse()

        else:
            print(f"{self.iter_method} method does not requires trust scores")

    # =========================================================================#
    #                       Some additional functionality                     #
    # =========================================================================#

    def get_avg_grad(self):
        avg_grad = OrderedDict(
            {
                key: torch.zeros_like(value, dtype=torch.float32)
                for key, value in self.server.client_gradients[0].items()
            }
        )
        for i in range(len(self.server.client_gradients)):
            for key, value in self.server.client_gradients[i].items():
                avg_grad[key] += value / float(self.num_clients)

        return avg_grad

    def get_scalar_prod(self, first, second):
        return torch.sum(first * second)

    def get_clients(self):
        bernoulli_dist = torch.distributions.Bernoulli(probs=self.q_m)
        self.probs = bernoulli_dist.sample((self.num_clients,))
        print(f"now we have selected clients: {self.probs}")
    
    def get_communication_content(self, rank):
        # In fedavg we need to send model after aggregate
        content = {
            "update_model": {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            },
        }
        if self.method == "scaffold":
            content["controls"] = (self.global_control, self.clients_control[rank])

        if self.server.error_feedback:
            content["do_update_error"] = (self.iter_compress_politic[rank] > 0)
            if (self.current_iter == self.iterations - 1) and ("ppbc" in self.method) and (self.server.error_feedback == "EF"):
                content["drop_error"] = True
        return content

    # =========================================================================#
    #                           Compressor Utilities                          #
    # =========================================================================#
    def random_compressor(self, mode="epoch"):
        if mode == "epoch":
            clients = np.arange(self.num_clients)
            random.shuffle(clients)

            self.epoch_compress_politic = torch.zeros_like(self.current_politic)
            for rank in range(self.epoch_k):
                self.epoch_compress_politic[clients[rank]] = self.current_politic[
                    clients[rank]
                ]

            print(
                clients,
                self.epoch_compress_politic,
                "perm of clients and politic for epoch",
            )

        if mode == "iter":
            clients = torch.arange(self.num_clients)
            nonzero_ranks = list(
                torch.nonzero(self.epoch_compress_politic.cpu(), as_tuple=True)[0]
            )
            random.shuffle(nonzero_ranks)

            self.iter_compress_politic = torch.zeros_like(self.epoch_compress_politic)
            for i in range(self.iter_k):
                self.iter_compress_politic[
                    nonzero_ranks[i]
                ] = self.epoch_compress_politic[nonzero_ranks[i]]
            print(
                nonzero_ranks,
                self.iter_compress_politic,
                "perm of clients and politic for iter",
            )
    
    def softmax_compressor(self, mode="epoch"):
        if mode == "epoch":
            p = softmax(self.epoch_prev_trust_scores)
            if np.isnan(p).any():
                p = np.ones(self.num_clients) / self.num_clients
            idx_of_k_clients = np.random.choice(np.arange(self.num_clients), replace=False, 
                                                size=(self.epoch_k,), p=p)

            self.epoch_compress_politic = torch.zeros_like(self.current_politic)
            for rank in range(self.epoch_k):
                self.epoch_compress_politic[
                    idx_of_k_clients[rank]
                ] = self.current_politic[idx_of_k_clients[rank]]

            print(
                self.epoch_prev_trust_scores,
                self.epoch_compress_politic,
                f"trust scores via {self.epoch_method} of clients and politic for epoch",
            )

        if mode == "iter":
            idx_of_k_clients = np.random.choice(np.arange(self.num_clients), replace=False, 
                                                size=(self.iter_k,), p=softmax(self.iter_prev_trust_scores))
            nonzero_rank = np.nonzero(self.epoch_compress_politic.cpu())
            best_epoch_results = idx_of_k_clients[
                np.isin(idx_of_k_clients, nonzero_rank)
            ]

            self.iter_compress_politic = torch.zeros_like(self.epoch_compress_politic)
            for rank in range(self.iter_k):
                self.iter_compress_politic[
                    best_epoch_results[rank]
                ] = self.epoch_compress_politic[best_epoch_results[rank]]

            print(
                self.iter_prev_trust_scores,
                self.iter_compress_politic,
                f"trust scores via {self.iter_method} of clients and politic for iter",
                flush=True,
            )


    def trust_score_compressor(self, mode="epoch"):
        if mode == "epoch":
            print(
                "trust scores1",
                self.epoch_prev_trust_scores,
            )
            idx_of_k_clients = np.argsort(self.epoch_prev_trust_scores)[::-1][
                : self.epoch_k
            ]

            self.epoch_compress_politic = torch.zeros_like(self.current_politic)
            for rank in range(self.epoch_k):
                self.epoch_compress_politic[
                    idx_of_k_clients[rank]
                ] = self.current_politic[idx_of_k_clients[rank]]

            print(
                self.epoch_prev_trust_scores,
                self.epoch_compress_politic,
                f"trust scores via {self.epoch_method} of clients and politic for epoch",
            )

        if mode == "iter":
            idx_of_k_clients = np.argsort(self.iter_prev_trust_scores)[::-1]
            nonzero_rank = np.nonzero(self.epoch_compress_politic.cpu())
            best_epoch_results = idx_of_k_clients[
                np.isin(idx_of_k_clients, nonzero_rank)
            ]

            self.iter_compress_politic = torch.zeros_like(self.epoch_compress_politic)
            for rank in range(self.iter_k):
                self.iter_compress_politic[
                    best_epoch_results[rank]
                ] = self.epoch_compress_politic[best_epoch_results[rank]]

            print(
                self.iter_prev_trust_scores,
                self.iter_compress_politic,
                f"trust scores via {self.iter_method} of clients and politic for iter",
                flush=True,
            )

    def epoch_compressor(self):
        if "random" in self.epoch_method:
            self.random_compressor(mode="epoch")
        elif "softmax" in self.epoch_method:
            self.softmax_compressor(mode="epoch")
        else:
            self.trust_score_compressor(mode="epoch")

    def iter_compressor(self):
        if "random" in self.iter_method:
            self.random_compressor(mode="iter")
        elif "softmax" in self.iter_method:
            self.softmax_compressor(mode="iter")
        else:
            self.trust_score_compressor(mode="iter")

    # =========================================================================#
    #                       Main algorithm functionality                      #
    # =========================================================================#

    def get_init_point(self):
        aggregated_weights = self.server.global_model.state_dict()
        for rank in range(self.num_clients):
            client_errors = self.final_errors[f"client {rank}"]

            for key, _ in aggregated_weights.items():
                aggregated_weights[key] = _ + client_errors[key]

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
            if self.server.error_feedback:
                client_approx_grad = self.server.client_approx_gradients[rank]
            else:
                client_approx_grad = client_grad
            current_client_error = self.current_errors_from_clients[f"client {rank}"]
            current_client_prob = self.probs[rank]
            final_client_error = self.final_errors[f"client {rank}"]
            client_politic = self.iter_compress_politic[rank].to(self.server.device)

            if self.method == "ppbc":
                for key, grads in client_grad.items():
                    self.current_errors_from_clients[f"client {rank}"][key] = (
                        current_client_error[key]
                        + (1 - self.theta)
                        * (1 / self.num_clients - client_politic)
                        * grads.to(self.server.device)
                        * int(self.need_errors)
                        * current_client_prob
                        / self.q_m
                    )
            elif self.method == "sppbc":
                for key, grads in client_approx_grad.items():
                    self.current_errors_from_clients[f"client {rank}"][key] = (
                    + (1 - self.theta)
                    * (1 / self.num_clients - client_politic)
                    * grads.to(self.server.device)
                    * int(self.need_errors)
                    * current_client_prob
                    / self.q_m
                )
            for key, grads in client_approx_grad.items():
                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + self.gamma
                    * (1 - self.theta)
                    * grads.to(self.server.device)
                    * client_politic
                    * int(self.need_errors)
                    * current_client_prob
                    / self.q_m
                    + self.gamma
                    * self.theta
                    * final_client_error[key]
                    * int(self.need_errors)
                    + self.gamma
                    * (1 - self.theta)
                    * grads.to(self.server.device)
                    * client_politic
                    * int(not self.need_errors)
                    #* (self.distribution[rank] / data_size)
                )
            if self.need_errors:
                if itn == self.iterations - 1:
                    self.final_errors[
                        f"client {rank}"
                    ] = self.current_errors_from_clients[f"client {rank}"]
                    print("final errors saved!")
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

        if self.need_errors:
            self.get_init_point()
        self.get_clients()
        for itn in range(self.iterations):
            self.current_iter = itn
            self.iter_compressor()
            print(f"start the {itn} iteration,")
            super().train_round()
            if self.method == "scaffold":
                aggregated_weights = self.get_scaffold_aggregation()
            else:
                aggregated_weights = self.get_errors_on_iter(itn)

            self._iter_count_trust_score()

            if (itn == self.iterations - 1) and ("bant" in self.epoch_method):
                self._epoch_count_trust_score()

            self.server.global_model.load_state_dict(aggregated_weights)
            print("nans in weights:", sum(v.isnan().any() for k,v in aggregated_weights.items()))

            print("processing is done")
        if "bant" not in self.epoch_method:
            self._epoch_count_trust_score()

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
        if self.method == "scaffold":
            self._init_controls()

        for round in range(self.rounds):
            self.round = round
            print(f"\nRound number: {round} of {self.rounds}")
            begin_round_time = time.time()
            self.cur_round = round

            _ = self.server.test_global_model()

            print("\nTraining started\n")

            self.process_clients()

            self.server.save_best_model(round)

            self.check_final_errors()

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        self.stop_train()
