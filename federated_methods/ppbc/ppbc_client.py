import copy
import torch

from ..base.client import Client, Top_K
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
    
    def init_error_feedback(self, cfg):
        self.momentum = {k: torch.zeros_like(v).to("cpu") for k,v in self.model.named_parameters()}
        self.updated_control = copy.deepcopy(self.momentum)
        if self.error_feedback:
            self.beta = torch.tensor(0.2, device="cpu")
            self.update_error = True
            compressor_name = cfg.training_params.compressor
            if "top" in compressor_name:
                self.compressor = Top_K(int(compressor_name[3:]), updated_params=dict(self.model.named_parameters()).keys())
            else:
                print(f"Compressor {compressor_name} is not supported")
        else:
            self.beta = torch.tensor(1.0, device="cpu")

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

        for k in self.local_control.keys():
            result_dict["delta_control"][k] = (
                self.updated_control[k].to(self.device) - self.local_control[k].to(self.device)
            )
            result_dict["client_control"][k] = self.local_control[k]

        if self.error_feedback:
            result_dict["delta_control"] = self.compressor(result_dict["delta_control"])

        return result_dict

    def _update_local_control(self, state):
        if not self.update_error:
            return
        # Option II in (4) from original paper https://arxiv.org/pdf/1910.06378.pdf

        # state = self.grad = local_model - global_model
        # -> coef with (-1)

        coef = -1 / (self.cfg.federated_params.round_epochs * self.lr)
        for k, v in self.local_control.items():
            self.momentum[k] = (
                (1 - self.beta) * self.momentum[k]
                + self.beta * (self.local_control[k] - self.global_control[k] + coef * state[k])
            )
        self.updated_control = {k:v.to("cpu") for k,v in self.momentum.items()}

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
        print(f"update error {self.update_error}")
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

    def get_grad(self):
        self.model.eval()
        for key, _ in self.model.state_dict().items():
                self.grad[key] = self.model.state_dict()[key].to(
                    "cpu"
                ) - self.server_model_state[key].to("cpu")