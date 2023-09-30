import json
import os
import argparse
from pathlib import Path

import optuna
from names_generator import generate_name
from omegaconf import OmegaConf

from .tune import run, WORKER_ID
from .utils import get_storage, load_config


def action_ls(storage, exec_name):
    studies = optuna.get_all_study_summaries(storage, False)
    studies_info = [(study.study_name, study.datetime_start) 
                    for study in studies if exec_name is None or study.study_name.startswith(exec_name)]

    for i, (name, starttime) in enumerate(studies_info):
                print(f"{i}.\t", name.ljust(64), starttime)


def action_rm(storage, exec_name):
    from intspan import intspan

    studies = optuna.get_all_study_summaries(storage, False)
    studies_info = [(study.study_name, study.datetime_start) 
                    for study in studies if exec_name is None or study.study_name.startswith(exec_name)]

    studies_to_delete = intspan(args.rm)
    print("You are deleting the following studies:")
    for study_i in studies_to_delete:
        print(study_i, studies_info[study_i][0])
    c = input("\nContinue? [y/n] ")
    if c in ["y", "Y"]:
        for study_i in studies_to_delete:
            optuna.delete_study(study_name=studies_info[study_i][0], storage=storage)


if __name__ == "__main__":
    try:
        with open(".stune/config.json", "r") as f:
            env = json.load(f)

            # Add current conda env if any is active
            env["CONDA_ENV"] = os.environ.get("CONDA_DEFAULT_ENV", None)
    except FileNotFoundError:
        print("stune hasn't been configured. Run 'python -m stune.config'.")

        exit(1)

    # Parse cmd line arguments
    parser = argparse.ArgumentParser(description="Slurm parallel hyperparameter optimization via optuna and neptune.")
    parser.add_argument("exec", nargs="?", type=str, help="Target python executable")
    parser.add_argument("--storage", type=str, help="URL of the storage used to save the study (if not specified, uses the configured psql server)")
    parser.add_argument("-s", "--study", type=str, help="Name of the study to create (or load if it already exists)")
    parser.add_argument("-t", "--n_trials", type=int, help="Number of trials to run the optimization for (exclusive with n_minutes)")
    parser.add_argument("-m", "--n_minutes", type=int, help="Number of minutes to run the optimization for (exclusive with n_trials)")
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
    parser.add_argument("--rm", type=str, help="List of studies to delete")

    args = parser.parse_args()
    args.exec = args.exec.replace(".py", "") if args.exec else None

    storage = get_storage(args, env)

    if args.ls:
        action_ls(storage, args.exec)
    elif args.rm:
        action_rm(storage, args.exec)
    else:
        # Compute study_name and exec_name by removing all unnecessary extensions and parent dirs
        exec_name = Path(args.exec).stem
        study_name = Path(args.study).name.replace(".yaml", "") or generate_name()

        # Check execution parameters
        if args.n_minutes is None and args.n_trials is None:
            args.n_trials = 1

        # Create config file if it is not worker
        config_name = f".stune/config/{exec_name}.{study_name}.cfg"
        if args.n_jobs != WORKER_ID:
            # Uses the raw args as they may contain the path to the files
            config = load_config(args.exec, args.study, args.config)
            OmegaConf.save(config, config_name)

        try:
            run(
                env,
                exec_name=exec_name,
                config_name=config_name,
                study_name=study_name,
                storage=storage,
                n_trials=args.n_trials,
                n_mins=args.n_minutes,
                sampler=args.sampler,
                debug=args.debug,
                log_level=args.log,
                n_jobs=args.n_jobs,
                partition=args.partition,
                description=args.msg
            )
        finally:
            if args.n_jobs != WORKER_ID and args.debug is True:
                os.remove(config_name)

