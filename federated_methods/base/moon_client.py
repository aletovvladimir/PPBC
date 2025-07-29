from .client import Client
from utils.model_utils import get_model
import torch

class MoonClient(Client):
    def create_pipe_commands(self):
        # define a structure to process pipe values
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map['get_grads'] = self.set_prev_model

        return pipe_commands_map
    
    def set_prev_model(self, prev_grad):
        self.prev_model = get_model(self.cfg)
        aggr_weights = self.model.state_dict()
        for key, value in prev_grad.items():
            aggr_weights[key] = aggr_weights[key].cpu() + value
            
        self.prev_model.load_state_dict(aggr_weights)
        
    def get_second_loss_val(self, inp):
        z = self.model(inp)
        z_glob = self.server_model(inp)
        z_prev = self.prev_model(inp)
        den = torch.exp(self.sim(z, z_glob)) + torch.exp(self.sim(z, z_prev))
        chisl = torch.exp(self.sim(z, z_glob))
        loss = -torch.log(chisl/den)
        return loss.mean() 
    
    def sim(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:

        # Приведение к одному типу и устройству
        tensor1 = tensor1.float()
        tensor2 = tensor2.float()
        
        # Добавим eps для предотвращения деления на ноль
        eps = 1e-8
        dot_product = torch.sum(tensor1 * tensor2, dim=-1)
        norm1 = torch.norm(tensor1, dim=-1)
        norm2 = torch.norm(tensor2, dim=-1)
        
        similarity = dot_product / (norm1 * norm2 + eps)
        return similarity
        
    
    def train_fn(self):
        self.server_model = get_model(self.cfg)
        self.server_model.load_state_dict(self.server_model_state)
        self.server_model.to(self.device)
        self.server_model.eval()
        self.prev_model.eval()
        self.prev_model.to(self.device)
        self.model.train()
        for _ in range(self.cfg.federated_params.round_epochs):
            for batch in self.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(outputs, targets)
                loss = loss + 1e-3 * self.get_second_loss_val(inp)

                loss.backward()

                self.optimizer.step()

                inp = input[0].to("cpu")
                targets = targets.to("cpu")