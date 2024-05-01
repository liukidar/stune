import os
import argparse

from omegacli import OmegaConf
import optuna

from .study import Storage, Study
from .tuner import Tuner
from .utils import load_config


def action_info(storage: Storage, exec_name):
    studies_info = action_ls(storage, exec_name)

    study = int(input("Study to load: "))
    study = optuna.study.load_study(study_name=studies_info[study][0], storage=storage.get())

    # Print running and failed trials
    print("Running trials:")
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.RUNNING:
            print(trial.number, trial.params)

    print("Failed trials:")
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.FAIL:
            print(trial.number, trial.params)


def action_ls(storage: Storage, exec_name):
    studies = optuna.get_all_study_summaries(storage.get(), False)
    studies_info = [
        (study.study_name, study.datetime_start)
        for study in studies if exec_name is None or study.study_name.startswith(exec_name)
    ]

    for i, (name, starttime) in enumerate(studies_info):
        print(f"{i}.\t", name.ljust(64), starttime)

    return studies_info


def action_rm(storage: Storage, exec_name):
    from intspan import intspan
    studies_info = action_ls(storage, exec_name)

    rm = input("Studies to delete: ")
    studies_to_delete = intspan(rm)
    print("You are deleting the following studies:")
    for study_i in studies_to_delete:
        print(study_i, studies_info[study_i][0])
    c = input("\nContinue? [y/n] ")
    if c in ["y", "Y"]:
        for study_i in studies_to_delete:
            optuna.delete_study(study_name=studies_info[study_i][0], storage=storage.get())


if __name__ == "__main__":
    # stune works by creating a configuration file which can be used to launch optimization workers.
    # Here we create the configuration file.
    config = OmegaConf.create({})

    # Load the stune configuration file
    try:
        config = OmegaConf.unsafe_merge(config, OmegaConf.load(".stune/config.yaml"))
        config["CONDA_ENV"] = os.environ.get("CONDA_DEFAULT_ENV", None)
    except FileNotFoundError:
        print("stune hasn't been configured in this folder. Run 'python -m stune.config'.")

        exit(1)

    # Create cmd line arguments parser
    parser = argparse.ArgumentParser(
        description="Configurable hyperparameter optimization via optuna. Supports SLURM scheduling."
    )

    # Add arguments

    # If the current process is a worker, it will be called with the --from_config argument
    parser.add_argument("--from_config", type=str, help="Run optimization from a configuration file")

    # else it is a scheduler and will be called with the following arguments to set up the optimization
    parser.add_argument("--exe", type=str, help="Target python script")
    parser.add_argument("--tuner", type=str, help="Tuner to use: ssh|slurm")
    parser.add_argument("--study", type=str, help="Name of the study in the format: 'experiment_name[study_name]'")
    parser.add_argument("--storage", type=str, help="URL of the storage host")
    parser.add_argument("--n_tasks", type=int, help="Number of tasks that are scheduled to be executed."
                        "By default enough tasks are scheduled to fill up available resources")
    parser.add_argument(
        "--n_trials",
        type=str,
        help="Number of trials to run the optimization for (exclusive with 'timeout')."
             "Format: 'n_trials:n_trials_per_worker'."
    )
    parser.add_argument(
        "--timeout",
        type=str,
        help="Number of minutes to run the optimization for (exclusive with 'n_trials')"
             "Format: 'timeout:timeout_per_worker'."
    )
    parser.add_argument("--sampler", type=str, help="Sampler used by optuna: tpe(None)|random|grid")
    parser.add_argument("--txt", type=str, help="Study description")

    # Override config
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the .yaml file to use to override the default configuration files"
    )

    # SBATCH exclusive arguments
    parser.add_argument("--n_jobs", type=int, help="[SLURM] Number of jobs that are scheduled to be executed")
    parser.add_argument("--partition", type=str, help="[SLURM] SLURM Partition to target when scheduling jobs")

    # SSH exclusive arguments
    parser.add_argument("--gpus", type=str, help="[SSH] List of GPUs to use")

    # STUNE exclusive arguments
    parser.add_argument(
        "--ls",
        action="store_true",
        help="List all studies. If exe is specified list only the studies on it."
    )
    parser.add_argument(
        "--rm",
        action="store_true",
        help="List all studies and ask for deletion. If exe is specified list only the studies on it."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="List all studies and ask for study to display. If exe is specified list only the studies on it."
    )

    # Load cli configuration
    cli_config, def_config = OmegaConf.from_argparse(parser)

    if (cfg_file := cli_config.get("from_config")) is not None:
        config = OmegaConf.unsafe_merge(
            OmegaConf.load(cfg_file),
            def_config,
            cli_config,
            config,
        )

        is_worker = True
    else:
        # Load experiment configuration
        exe_config = load_config(cli_config.get("exe"))

        # Load study configuration
        study_config = load_config(cli_config.get("study"))

        # Load custom configuration
        custom_config = load_config(cli_config.get("config"))

        config = OmegaConf.unsafe_merge(
            def_config,
            config,
            exe_config,
            study_config,
            custom_config,
            cli_config,
        )

        is_worker = False

    storage = Storage.from_config(config)

    if config.ls:
        action_ls(storage, config.get("exe", None))
    elif config.rm:
        action_rm(storage, config.get("exe", None))
    elif config.info:
        action_info(storage, config.get("exe", None))
    else:
        study = Study.from_config(config)
        tuner = Tuner.from_config(config)
        # Create config file if it is not worker
        config_name = f".stune/configs/{study.name}.cfg"
        if not is_worker:
            OmegaConf.save(config, config_name)

            tuner.schedule(study, storage)
        else:
            tuner.run(study, storage)
