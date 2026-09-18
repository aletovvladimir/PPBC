from ..base.fedavg import FedAvg
from ..fedprox.fedprox_client import FedProxClient

class FedProx_epoch(FedAvg):
    def __init__(self, fed_prox_lambda, num_fedavg_rounds, **method_args):
        super().__init__()

        # FedProx params
        self.fed_prox_lambda = fed_prox_lambda
        self.num_fedavg_rounds = num_fedavg_rounds

        # Epoch-level trust/compression params (PPBC-style)
        self.epoch_method = method_args.get("epoch_method", "angle")
        self.epoch_k = method_args.get("epoch_k", 3)

        # Iteration-level sub-selection params (PPBC-style)
        self.iter_method = method_args.get("iter_method", "random")
        self.iter_k = method_args.get("iter_k", 1)
        self.iterations = method_args.get("iterations", 1)

        self.q_m = method_args.get("q_m", 1.0)
        self.strategy = method_args.get("strategy", "top")
        self.gamma = method_args.get("gamma", 1.0)

        # Error-feedback params (PPBC-style)
        self.theta = method_args.get("theta", 0.0)
        self.need_errors = method_args.get("need_errors", False)

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

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

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.num_clients = cfg.federated_params.amount_of_clients
        self.epoch_prev_trust_scores = [1 / self.num_clients] * self.num_clients
        self.iter_prev_trust_scores = [1 / self.num_clients] * self.num_clients

    # =========================================================================#
    #                          FedProx client wiring                           #
    # =========================================================================#

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedProxClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.fed_prox_lambda, self.num_fedavg_rounds])

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["current_round"] = self.cur_round
        return content

    # =========================================================================#
    #                    Trust score (angle) — PPBC logic                     #
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

    def get_score_from_angle(self):
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

    def _epoch_count_trust_score(self):
        if "angle" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_score_from_angle()
        else:
            print(f"{self.epoch_method} method does not require trust scores")

    def _iter_count_trust_score(self):
        if "angle" in self.iter_method:
            self.iter_prev_trust_scores = self.get_score_from_angle()
        else:
            print(f"{self.iter_method} method does not require trust scores")

    # =========================================================================#
    #                          Compressor utilities                           #
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

    def trust_score_compressor(self, mode="epoch"):
        if mode == "epoch":
            if self.strategy == "top":
                idx_of_k_clients = np.argsort(self.epoch_prev_trust_scores)[::-1][
                    : self.epoch_k
                ]
            elif self.strategy == "sample":
                scores = np.array(self.epoch_prev_trust_scores)
                idx_of_k_clients = np.random.choice(
                    self.num_clients,
                    p=scores / scores.sum(),
                    replace=False,
                    size=self.epoch_k,
                )
            else:
                raise ValueError("not correct strategy!")

            self.epoch_compress_politic = torch.zeros_like(self.current_politic)
            for rank in idx_of_k_clients:
                self.epoch_compress_politic[rank] = self.current_politic[rank]

            print(
                self.epoch_prev_trust_scores,
                self.epoch_compress_politic,
                f"trust scores via {self.epoch_method} of clients and politic for epoch",
            )

        if mode == "iter":
            nonzero_rank = np.array(self.epoch_compress_politic.cpu()).nonzero()[0]

            if self.strategy == "top":
                idx_of_k_clients = np.argsort(self.iter_prev_trust_scores)[::-1]
                best_epoch_results = idx_of_k_clients[
                    np.isin(idx_of_k_clients, nonzero_rank)
                ]
            elif self.strategy == "sample":
                p = np.array(self.iter_prev_trust_scores)[nonzero_rank]
                p = p / p.sum()
                best_epoch_results = np.random.choice(
                    nonzero_rank, p=p, replace=False, size=self.iter_k
                )
            else:
                raise ValueError("not correct strategy!")

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
        else:
            self.trust_score_compressor(mode="epoch")

    def iter_compressor(self):
        if "random" in self.iter_method:
            self.random_compressor(mode="iter")
        else:
            self.trust_score_compressor(mode="iter")

    # =========================================================================#
    #                     Client sampling — PPBC logic                        #
    # =========================================================================#

    def get_clients(self):
        bernoulli_dist = torch.distributions.Bernoulli(probs=self.q_m)
        self.probs = bernoulli_dist.sample((self.num_clients,))
        print(f"now we have selected clients: {self.probs}")

    # =========================================================================#
    #                Aggregation — PPBC logic (error-feedback aware)          #
    # =========================================================================#

    def get_data_size(self):
        current_data_size = torch.sum(
            self.iter_compress_politic
            * torch.tensor(self.distribution).to(self.server.device)
            * self.num_clients
        )
        return current_data_size

    def get_init_point(self):
        aggregated_weights = self.server.global_model.state_dict()
        for rank in range(self.num_clients):
            client_errors = self.final_errors[f"client {rank}"]
            for key, val in aggregated_weights.items():
                aggregated_weights[key] = val + client_errors[key] * 0.5

        for key, val in aggregated_weights.items():
            if "running_var" in key:
                aggregated_weights[key] = torch.clamp(val, min=0.01)

        self.server.global_model.load_state_dict(aggregated_weights)

    def init_errors(self):
        # zero out current-round error accumulators; seed final_errors on round 0
        for rank in range(self.num_clients):
            for key, val in self.server.global_model.state_dict().items():
                self.current_errors_from_clients[f"client {rank}"][key] = (
                    torch.zeros_like(val).to(self.server.device)
                )
                if self.cur_round == 0:
                    self.final_errors[f"client {rank}"][key] = torch.zeros_like(
                        val
                    ).to(self.server.device)

    def get_errors_on_iter(self, itn):
        aggregated_weights = self.server.global_model.state_dict()

        data_size = self.get_data_size()
        print(f"now we use {data_size} objects from dataset")

        for rank in range(self.num_clients):
            client_grad = self.server.client_gradients[rank]
            current_client_error = self.current_errors_from_clients[f"client {rank}"]
            current_client_prob = self.probs[rank]
            final_client_error = self.final_errors[f"client {rank}"]
            client_politic = self.iter_compress_politic[rank].to(self.server.device)

            for key, grads in client_grad.items():
                if self.need_errors:
                    self.current_errors_from_clients[f"client {rank}"][key] = (
                        current_client_error[key]
                        + (1 - self.theta)
                        * (1 / self.num_clients - client_politic)
                        * grads.to(self.server.device)
                        * current_client_prob
                        / self.q_m
                    )

                    aggregated_weights[key] = (
                        aggregated_weights[key]
                        + self.gamma
                        * (1 - self.theta)
                        * grads.to(self.server.device)
                        * client_politic
                        * current_client_prob
                        / self.q_m
                        + self.gamma
                        * self.theta
                        * final_client_error[key]
                        * 0.5
                    )
                else:
                    aggregated_weights[key] = (
                        aggregated_weights[key]
                        + self.gamma
                        * grads.to(self.server.device)
                        * client_politic
                        * current_client_prob
                        * (self.distribution[rank] / data_size)
                    )

            if self.need_errors:
                if itn == self.iterations - 1:
                    self.final_errors[
                        f"client {rank}"
                    ] = self.current_errors_from_clients[f"client {rank}"]
                    print("final errors saved!")

        return aggregated_weights

    # =========================================================================#
    #                       Main algorithm functionality                      #
    # =========================================================================#

    def init_politic(self):
        self.current_politic = (
            torch.ones(self.num_clients).to(self.server.device) / self.num_clients
        )

    def process_clients(self):
        if self.cur_round == 0:
            self.init_politic()

        self.epoch_compressor()

        if self.need_errors:
            self.init_errors()
            self.get_init_point()

        self.get_clients()

        for itn in range(self.iterations):
            self.iter_compressor()
            print(f"start the {itn} iteration")

            super().train_round()

            aggregated_weights = self.get_errors_on_iter(itn)

            self._iter_count_trust_score()

            for key, val in aggregated_weights.items():
                if "running_var" in key:
                    aggregated_weights[key] = torch.clamp(val, min=0.01)
            self.server.global_model.load_state_dict(aggregated_weights)

            print("processing is done")

        self._epoch_count_trust_score()

    def check_final_errors(self):
        if not self.need_errors:
            return
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
