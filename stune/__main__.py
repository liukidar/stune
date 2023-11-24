import json
import os
import argparse
from pathlib import Path
import enum

from names_generator import generate_name
from omegaconf import OmegaConf
import optuna

from .tune import run
from .utils import Study, Storage, load_config


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
    studies_info = [(study.study_name, study.datetime_start) 
                    for study in studies if exec_name is None or study.study_name.startswith(exec_name)]

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
    # Check if stune is configured in the current environment
    try:
        with open(".stune/config.json", "r") as f:
            env = json.load(f)

            # Add current conda env if any is active
            env["CONDA_ENV"] = os.environ.get("CONDA_DEFAULT_ENV", None)
    except FileNotFoundError:
        print("stune hasn't been configured in this folder. Run 'python -m stune.config'.")

        exit(1)

    # Parse cmd line arguments
    parser = argparse.ArgumentParser(description="Slurm parallel hyperparameter optimization via optuna and neptune.")
    parser.add_argument("exec", nargs="?", type=str, help="Target python executable")
    parser.add_argument("--storage", type=str, help="URL of the storage host")
    parser.add_argument("-s", "--study", type=str, help="Name of the study to create (or load if it already exists)")
    parser.add_argument("-t", "--n_trials", type=str, help="Number of trials to run the optimization for (exclusive with n_minutes)")
    parser.add_argument("--sampler", type=str, help="Sampler used by optuna: tpe(None)|random|grid")
    parser.add_argument("-d", "--debug", action="store_true", help="Whether to run the optimization in debug mode")
    parser.add_argument("-l", "--log", type=str, help="Log level: None|trial|study|all")
    parser.add_argument("--msg", type=str, help="Study description")
    
    # Override config
    parser.add_argument(
        "--config", type=str, help="Name of the .yaml file to use to override the default configuration files", default=None
    )

    # SBATCH exclusive arguments
    parser.add_argument("-j", "--n_jobs", type=int, default=0, help="Number of jobs that are scheduled to be executed. \
        If n_jobs is 0 (default) then the optimization is immediately run on the active node, otherwise it is scheduled via sbatch.")
    parser.add_argument("--partition", type=str, help="SLURM Partition to target when scheduling jobs", default="small")
    
    # OPTUNA exclusive arguments
    parser.add_argument("--ls", action="store_true", help="List all studies. If exec is specified list only the studies on it.")
    parser.add_argument("--rm", action="store_true", help="List all studies and ask for deletion. If exec is specified list only the studies on it.")
    parser.add_argument("--info", action="store_true", help="List all studies and ask for study to display. If exec is specified list only the studies on it.")

    # Reserved arguments
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    args.exec = args.exec.replace(".py", "") if args.exec else None

    storage = Storage.init(args, env)

    if args.ls:
        action_ls(storage, args.exec)
    elif args.rm:
        action_rm(storage, args.exec)
    elif args.info:
        action_info(storage, args.exec)         
    else:
        # Compute study_name and exec_name by removing all unnecessary extensions and parent dirs
        exec_name = Path(args.exec).stem
        study_name = Path(args.study).name.replace(".yaml", "") or generate_name()
        study = Study.init(args, exec_name, study_name)

        # Create config file if it is not worker
        config_name = f".stune/config/{study.name}.cfg"
        if study.is_worker() == False:
            # Uses the raw args as they may contain the path to the files
            config = load_config(args.exec, args.study, args.config)
            OmegaConf.save(config, config_name)

        try:
            run(
                env,
                study=study,
                storage=storage,
                config_name=config_name,
                debug=args.debug,
                log_level=args.log,
                description=args.msg
            )
        finally:
            if study.is_worker() == False and args.debug is True:
                os.remove(config_name)

