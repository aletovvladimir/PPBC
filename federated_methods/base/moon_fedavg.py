from .fedavg import FedAvg 

class MoonServer(FedAvg):
    def get_communication_content(self, rank):
        # In fedavg we need to send model after aggregate
        return {
            "update_model": {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            },
            'get_grads': self.server.client_gradients[rank] 
        }