import torch
from collections import OrderedDict

from ..base.server import Server
from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from utils.model_utils import get_model


class TSMomentumServer(Server):
    def __init__(self, cfg, trust_df):
        super().__init__(cfg)
        self.trust_df = trust_df
        self.trust_loader = get_dataset_loader(
            self.trust_df, cfg, drop_last=False, mode="train"
        )

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df,
        )

    def eval_trust_fn(self, model_weights):
        self.client_model = get_model(self.cfg)
        self.client_model.load_state_dict(model_weights)
        self.client_model.to(self.device)
        self.client_model.eval()
        self._init_criterion()

        loss = 0
        with torch.no_grad():
            for _, batch in enumerate(self.trust_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)

                targets = targets.to(self.device)
                outputs = self.client_model(inp)

                loss += self.criterion(outputs, targets)

        self.client_model.to("cpu")

        loss /= len(self.trust_loader) + int(
            bool(len(self.trust_df) % len(self.trust_loader))
        )

        return loss.cpu()

    def get_client_weights(self, client_gradients):
        client_weights = OrderedDict()
        for key, weight in self.global_model.state_dict().items():
            client_weights[key] = client_gradients[key].to(self.device) + weight
        return client_weights

    def get_trust_losses(self):
        trust_losses = []
        server_loss = self.eval_trust_fn(self.global_model.state_dict())
        print(f"Server trust loss: {server_loss}\n", flush=True)
        for i, client_gradients in enumerate(self.client_gradients):
            client_weights = self.get_client_weights(client_gradients)
            client_loss = self.eval_trust_fn(client_weights)
            print(f"Client {i} trust loss: {client_loss}", flush=True)
            trust_losses.append(client_loss)
        return server_loss, trust_losses
