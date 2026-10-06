from hydra.utils import instantiate


class Manager:
    def __init__(self, cfg, server) -> None:
        self.server = server
        self.cfg = cfg
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.batches = instantiate(
            cfg.manager, amount_of_clients=self.amount_of_clients
        )

    def step(self, batch_idx):
        # Step the manager to update ranks for clients
        current_batch = self.batches.get_batch(batch_idx)
        next_batch = self.batches.get_batch((batch_idx + 1) % len(self.batches))

        for client_idx, rank in enumerate(current_batch):
            new_rank = next_batch[client_idx]
            content = {"reinit": new_rank}
            self.server.send_content_to_client(client_idx, content)

    def get_clients_loader(self):
        return self.batches


class SequentialIterator:
    def __init__(self, batch_size, amount_of_clients):
        self.amount_of_clients = amount_of_clients
        self.ranks = [i for i in range(self.amount_of_clients)]
        self.batch_size = self.define_batch_len(batch_size)
        self.num_batches = len(self.ranks) // batch_size + (
            1 if len(self.ranks) % batch_size != 0 else 0
        )

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx < len(self.ranks):
            batch = self.ranks[self.current_idx : self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def define_batch_len(self, batch_size):
        if batch_size == "dynamic":
            # IMPLEMENT LATER
            assert (
                False
            ), "At the current moment we do not support dynamic size of processes batch"
        else:
            return min(self.amount_of_clients, batch_size)

    def get_batch(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.ranks[start:end]
