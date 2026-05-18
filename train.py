# Cap BLAS/OpenMP thread pools before numpy/torch/transformers load native libs.
_CPU_THREADS = "2"
for _env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_var, _CPU_THREADS)


import os
import hydra
import random
import signal
import torch
from functools import partial
from omegaconf import DictConfig
from hydra.utils import instantiate

from utils.data_utils import prepare_df_for_federated_training, set_up_base_dir
from utils.utils import handle_main_process_sigterm
from utils.logging_utils import redirect_stdout_to_log
from utils.dirichlet import DirichletDistribution

torch.set_num_threads(_NUM_CPU_THREADS)
torch.set_num_interop_threads(1)

# Make print with flush=True by default
print = partial(print, flush=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    redirect_stdout_to_log()
    cfg = set_up_base_dir(cfg)
    df, cfg = prepare_df_for_federated_training(cfg, "train_directories")

    distr = DirichletDistribution(alpha=cfg.dirichlet_alpha, verbose=True)

    df = distr.split_to_clients(df=df, amount_of_clients=cfg.federated_params.amount_of_clients, random_state=cfg.random_state)

    # Needed params for multiprocessing
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(random.randint(30000, 60000))
    # Init federated_method and begin train
    trainer = instantiate(cfg.federated_method, _recursive_=False)
    trainer._init_federated(cfg, df)
    # Termination handling in multiprocess setup
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_main_process_sigterm(signum, frame, trainer),
    )
    print("start federfated")
    trainer.begin_train()


if __name__ == "__main__":
    train()
