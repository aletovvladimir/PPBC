import copy
import torch

from ..base.client import Client
import torch
from hydra.utils import instantiate


class ScaffoldClient(Client):
    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]  # `cfg` , `df`
        self.lr = client_args[2]  # `local_lr`

        super().__init__(*base_client_args, **client_kwargs)

        # Need after super init
        self.client_args = client_args
        self.client_kwargs = client_kwargs

        self.local_control = None

    def _init_optimizer(self):
        # Maybe unnecessarry
        self.optimizer = instantiate(
            self.cfg.optimizer, params=self.model.parameters(), lr=self.lr
        )

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["controls"] = self.set_controls
        return pipe_commands_map

    def set_controls(self, controls):
        # controls = (global_c, client_c)
        self.global_control = controls[0]
        self.local_control = controls[1]

    def get_communication_content(self):
        # In scaffold_client we need to send result of local learning
        # and delta controls (c_+ - c_i)

        result_dict = super().get_communication_content()

        result_dict["delta_control"] = {}
        result_dict["client_control"] = {}

        for k in self.updated_control.keys():
            result_dict["delta_control"][k] = (
                self.updated_control[k] - self.local_control[k]
            )
            result_dict["client_control"][k] = self.updated_control[k]

        return result_dict

    def _update_local_control(self, state):
        # Option II in (4) from original paper https://arxiv.org/pdf/1910.06378.pdf

        # state = self.grad = local_model - global_model
        # -> coef with (-1)

        self.updated_control = {}

        coef = -1 / (self.cfg.federated_params.round_epochs * self.lr)
        for k, v in self.local_control.items():
            self.updated_control[k] = (
                self.local_control[k] - self.global_control[k] + coef * state[k]
            )

    def _add_grad_control(self):
        weights = copy.deepcopy(self.model.state_dict())

        with torch.no_grad():
            for k, v in self.grad_control.items():
                weights[k].add_(self.grad_control[k])

        self.model.load_state_dict(weights)

    def calculate_grad_control(self):
        grad_control = {}
        for k, v in self.model.named_parameters():
            grad_control[k] = self.lr * (
                self.local_control[k].data - self.global_control[k].data
            )
            grad_control[k] = grad_control[k].to(self.device)

        return grad_control

    def train(self):
        # Gradient control is addition to gradient in local learning in SCAFFOLD
        self.grad_control = self.calculate_grad_control()

        super().train()

        self._update_local_control(self.grad)

    def train_fn(self):
        self.model.train()
        for _ in range(self.cfg.federated_params.round_epochs):
            for batch in self.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(outputs, targets)

                loss.backward()

                self.optimizer.step()

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

                # ------SCAFFOLD------#
                self._add_grad_control()
                # ------SCAFFOLD------#
