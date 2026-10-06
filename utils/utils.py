import sys
import os
import signal
import subprocess
from hydra.core.hydra_config import HydraConfig


def handle_main_process_sigterm(signum, frame, trainer):
    """This function manage behaviour when main process is terminated by user

    Args:
        signum (int): number of signal
        frame (signal.frame): current stack frame
    """
    print("Parent process received SIGTERM. Terminate all child processes...")
    trainer.stop_train()
    sys.exit(0)


def handle_client_process_sigterm(signum, frame, rank):
    """This function manage behaviour when child process is terminated by user

    Args:
        signum (int): number of signal
        frame (signal.frame): current stack frame
    """
    print(f"Child process {rank} received SIGTERM. Send it to parent...")

    os.kill(os.getppid(), signal.SIGTERM)  # Send SIGTERM to parent process


def create_model_info(model_state, metrics, checkpoint_path, cfg):
    model_info = {
        "model": model_state,
        "metrics": {"metrics": metrics[0], "threshold": metrics[1]},
        "config_file": cfg,
    }
    filename = f"{HydraConfig.get().run.dir}/experiment_report.txt"
    with open(filename, "w") as f:
        f.write(
            f"- Model's checkpoint:\n_{HydraConfig.get().runtime.cwd}/{checkpoint_path}/_\n"
        )
        f.write("=============METRICS REPORT=============\n")
        f.write(
            "\n".join(
                [
                    line
                    for i, line in enumerate(
                        model_info["metrics"]["metrics"]
                        .to_markdown(index=True)
                        .split("\n")
                    )
                    if i != 1
                ]
            )
        )

    return model_info
