import json
import os
import argparse

import optuna
from names_generator import generate_name
from omegaconf import OmegaConf

from .tune import run, WORKER_ID


def get_storage(args):
    storage = args.storage
    if args.debug is False and storage is None:
        storage = f"postgresql://{os.environ['PSQL_USR']}:{os.environ['PSQL_PWD']}@{os.environ['PSQL_HOST']}"

    return storage


def action_ls(storage):
    studies = optuna.get_all_study_summaries(storage, False)
    studies_info = [(study.study_name, study.datetime_start) 
                    for study in studies if args.exec is None or study.study_name.startswith(args.exec)]

    for i, (name, starttime) in enumerate(studies_info):
                print(f"{i}.\t", name.ljust(64), starttime)


def action_rm(storage):
    from intspan import intspan

    studies = optuna.get_all_study_summaries(storage, False)
    studies_info = [(study.study_name, study.datetime_start) 
                    for study in studies if args.exec is None or study.study_name.startswith(args.exec)]

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
            for key, value in env.items():
                os.environ[key] = value
    except FileNotFoundError:
        print("stune hasn't been configured. Run 'python -m stune.config'.")

        exit(1)

    # Parse cmd line arguments
    parser = argparse.ArgumentParser(description="Slurm parallel hyperparameter optimization via optuna and neptune.")
    parser.add_argument("exec", nargs=1, type=str, help="Target python executable")
    parser.add_argument("--storage", type=str, help="URL of the storage used to save the study")
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
    args.exec = args.exec[0].replace(".py", "")

    storage = get_storage(args)

    if args.ls is not None:
        action_ls(storage)
    elif args.rm is not None:
        action_rm(storage)
    else:
        # Compute study_name
        study_name = args.study or generate_name()

        # Check execution parameters
        if args.n_minutes is None and args.n_trials is None:
            args.n_trials = 1

        # Create config file if it is not worker
        config_name = f".stune/config/{args.exec}_{study_name}.cfg"
        if args.n_jobs != WORKER_ID:
            # Load exec configuration
            try:
                exec_config = OmegaConf.load(args.exec + ".yaml")
            except FileNotFoundError:
                exec_config = OmegaConf.create()

            # Load tune configuration
            try:
                tune_config = OmegaConf.load(args.exec + ".tune.yaml")
            except FileNotFoundError:
                tune_config = OmegaConf.create()

            # Load manual configuration
            manual_config = OmegaConf.load(args.config.replace(".yaml", "") + ".yaml") \
                            if args.config else OmegaConf.create()

            config = OmegaConf.unsafe_merge(
                exec_config,
                tune_config,
                manual_config
            )
            OmegaConf.save(config, config_name)

        run(
            exec_name=args.exec,
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

        if args.n_jobs != -1 and args.debug is True:
            os.remove(config_name)

