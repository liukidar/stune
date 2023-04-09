import os
from typing import Any, Optional
import importlib
from datetime import datetime
import argparse
import functools

import optuna
import optuna.visualization
import neptune.integrations.optuna as optuna_utils
from omegaconf import OmegaConf
from names_generator import generate_name

from .slurm import sbatch
from . import RunInfo, open_log


def print_time(str: Optional[str] = None) -> None:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', str)


def objective(
    trial: Any,
    study_name: str,
    exec: Any,
    params: OmegaConf,
    log_mode: Optional[str] = None
):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if log_mode is not None:
        project_name = os.path.basename(os.path.dirname(os.path.realpath(exec.__file__))).lower()
        exec_name = os.path.basename(exec.__file__).split(".")[0]
        with open_log(
            project_name,
            exec_name,
            mode=log_mode,
            sweep_id=study_name,
            level_tag="trial-level"
        ) as log:
            res = exec.main(RunInfo(params, log, trial))
    else:
        res = exec.main(RunInfo(params, None, trial))

    return res


def main(
    exec_name: str,
    study_name: str,
    storage: str,
    n_trials: int,
    n_mins: Optional[int] = None,
    debug: bool = False,
    log: bool = False,
    n_jobs: int = 0,
    partition: str = "devel",
    description: Optional[str] = None
):
    exec_config_name = f".stune/config/{exec_name}_{study_name}.cfg"
    exec_config = OmegaConf.load(exec_config_name)

    study = optuna.create_study(
        study_name=f"{exec_name}_{study_name}",
        storage=storage,
        load_if_exists=True,
        directions=exec_config.directions.values(),
        # sampler=optuna.samplers.RandomSampler()
    )
    
    exec = importlib.import_module(exec_name, )

    if n_jobs == 0:

        log_mode = None
        if log is True:
            log_mode = "debug" if debug is True else "offline"

        study.optimize(
            functools.partial(
                objective,
                study_name=f"{exec_name}_{study_name}",
                exec=exec,
                params=exec_config,
                log_mode=log_mode,
            ),
            n_trials=n_trials,
            timeout=n_mins,
            n_jobs=1
        )
    else:
        gpus_per_task = exec_config.gpus_per_task if debug is False else 1
        tasks_per_node = max(1, int(1 / gpus_per_task))
        cpus_per_task = exec_config.cpus_per_task

        requested_gpus = max(1, gpus_per_task)

        cmd = f"python -m tune.run -e {exec_name} --study {study_name} --storage {storage} --n_jobs 0 "
        if n_mins is not None:
            cmd += f"--n_minutes {n_mins * 60 - 2 * 60} "
        if n_trials is not None:
            cmd += f"--n_trials {n_trials}"

        if debug is True:
            cmd += "--debug " 

        if n_mins is not None or n_trials is not None:
            if n_mins is None:
                n_mins = 60
            
            sbatch(
                cmd,
                n_jobs=n_jobs,
                tasks_per_node=tasks_per_node,
                cpus_per_task=cpus_per_task,
                time_hours=n_mins // 60,
                time_minutes=n_mins % 60,
                job_name=f"{exec_name}_{study_name}", 
                gpus=requested_gpus,
                partition=partition if debug is False else 'devel',
                env="pcax",
                cuda=True,
                wait=True
            )

        if debug is not True:
            with open_log(
                os.path.basename(os.path.dirname(os.path.realpath(exec.__file__))).lower(),
                exec_name,
                description=description,
                mode="async",
                sweep_id=f"{exec_name}_{study_name}",
                level_tag="study-level",
                custom_run_id=f"{exec_name}_{study_name}",
            ) as log_study:
                optuna_utils.log_study_metadata(
                    study,
                    log_study,
                    target_names=tuple(exec_config.directions.keys()),
                    log_all_trials=False,
                    log_plot_optimization_history=False,
                    log_plot_edf=False,
                    log_plot_contour=False,
                    log_distributions=False
                )
                import matplotlib.pyplot as plt
                optuna.visualization.matplotlib.plot_contour(study)
                log_study["visualizations/plot_contour"].upload(plt.gcf())
                log_study["config"].upload(exec_config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slurm parallel hyperparameter optimization via optuna and neptune.")
    parser.add_argument("-e", "--exec", type=str, help="Target executable name (without extension)")
    parser.add_argument("--storage", type=str, help="URL of the storage used to save the study")
    parser.add_argument("-s", "--study", type=str, help="Name of the study to create (or load if it already exists)")
    parser.add_argument("-t", "--n_trials", type=int, help="Number of trials to run the optimization for (exclusive with n_minutes)")
    parser.add_argument("-m", "--n_minutes", type=int, help="Number of minutes to run the optimization for (exclusive with n_trials)")
    parser.add_argument("-d", "--debug", action="store_true", help="Whether to run the optimization in debug mode")
    parser.add_argument("-l", "--log", action="store_true", help="Whether to enable logging for each individual trial")
    parser.add_argument("--msg", type=str, help="Study description")
    
    # Override config
    parser.add_argument(
        "--config", type=str, help="Name of the .yaml file to use to override the default configuration files", default=None
    )

    # SBATCH exclusive arguments
    parser.add_argument("-j", "--n_jobs", type=int, default=1, help="Number of jobs that are scheduled to be executed. \
        If n_jobs is 0 (default is 1) then the optimization is immediately run on the active node, otherwise it is scheduled via sbatch.")
    parser.add_argument("--partition", type=str, help="SLURM Partition to target when scheduling jobs", default="small")
    
    # OPTUNA exclusive arguments
    parser.add_argument("--ls", action="store_true", help="List all studies. If exec is specified list only the studies on it.")
    parser.add_argument("--rm", type=str, help="List of studies to delete")

    args = parser.parse_args()


    # Compute storage
    storage = args.storage or f"postgresql://{os.environ['PSQL_USR']}:{os.environ['PSQL_PWD']}@{os.environ['PSQL_HOST']}"

    if args.ls or args.rm is not None:
        studies = optuna.get_all_study_summaries(storage, False)
        studies_info = [(study.study_name, study.datetime_start) for study in studies if args.exec is None or study.study_name.startswith(args.exec)]

        if args.ls:
            for i, (name, starttime) in enumerate(studies_info):
                print(f"{i}.\t", name.ljust(64), starttime)

        if args.rm is not None:
            from intspan import intspan

            studies_to_delete = intspan(args.rm)
            print("You are deleting the following studies:")
            for study_i in studies_to_delete:
                print(study_i, studies_info[study_i][0])
            c = input("Continue? [y/n]")
            if c == "y":
                for study_i in studies_to_delete:
                    optuna.delete_study(study_name=studies_info[study_i][0], storage=storage)
    else:
        # Compute study_name
        study_name = args.study or generate_name()

        # Create config file
        if args.study is None:
            exec_config_name = f".stune/config/{args.exec}_{study_name}.cfg"

            # Load exec configuration, including tuning and default tuning files.
            exec_config = OmegaConf.unsafe_merge(
                OmegaConf.load(".tune.yaml"),
                OmegaConf.load(args.exec + ".yaml"),
                OmegaConf.load(args.exec + ".tune.yaml"),
                OmegaConf.load(args.config)
                if args.config is not None
                else OmegaConf.create()
            )
            OmegaConf.save(exec_config, exec_config_name)

        main(
            exec_name=args.exec,
            study_name=study_name,
            storage=storage,
            n_trials=args.n_trials,
            n_mins=args.n_minutes,
            debug=args.debug,
            log=args.log,
            n_jobs=args.n_jobs,
            partition=args.partition,
            description=args.msg
        )
