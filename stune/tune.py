import os
from typing import Any, Optional
import importlib
from datetime import datetime
import functools

import optuna
import optuna.visualization
import neptune.integrations.optuna as optuna_utils
from omegaconf import OmegaConf

from .slurm import sbatch
from . import RunInfo, open_log


WORKER_ID = -1


def print_time(str: Optional[str] = None) -> None:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', str)


def objective(
    trial: Any,
    study_name: str,
    exec: Any,
    params: OmegaConf,
    log_mode: Optional[str] = None
):
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
            run_info = RunInfo(study_name, params, log, trial)
    else:
        run_info = RunInfo(study_name, params, None, trial)

    return exec.main(run_info)


def run(
    env,
    exec_name: str,
    config_name: str,
    study_name: str,
    storage: Optional[str],
    n_trials: int,
    n_mins: Optional[int] = None,
    sampler: Optional[str] = "random",
    debug: bool = False,
    log_level: Optional[str] = None,
    n_jobs: int = 0,
    partition: str = "devel",
    description: Optional[str] = None
):
    config = OmegaConf.load(config_name)

    samplers = {
        None: lambda: None,
        "random": optuna.samplers.RandomSampler,
        "grid": optuna.samplers.BruteForceSampler
    }

    study = optuna.create_study(
        study_name=f"{exec_name}.{study_name}",
        storage=storage,
        load_if_exists=True,
        directions=config.directions.values(),
        sampler=samplers[sampler]()
    )

    # Worker
    if n_jobs < 1:
        exec = importlib.import_module(exec_name)
        
        if log_level in ["trial", "all"]:
            log_mode = "debug" if debug is True else "offline"
        else:
            log_mode = None

        study.optimize(
            functools.partial(
                objective,
                study_name=f"{exec_name}.{study_name}",
                exec=exec,
                params=config,
                log_mode=log_mode,
            ),
            n_trials=n_trials,
            timeout=n_mins,
            n_jobs=1
        )
    # Scheduler
    else:
        gpus_per_task = config.gpus_per_task if debug is False else 1
        tasks_per_node = max(1, int(1 / gpus_per_task))
        cpus_per_task = config.cpus_per_task

        requested_gpus = max(1, gpus_per_task)

        cmd = (f"python -m stune {exec_name}"
               f" --study={study_name}"
               f" --storage={storage if storage is not None else ''}"
               f" --n_jobs='{WORKER_ID}'")
        if n_mins is not None:
            cmd += f" --n_minutes={n_mins * 60 - 2 * 60} "
        if n_trials is not None:
            cmd += f" --n_trials={n_trials}"

        if debug is True:
            cmd += " --debug" 

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
                job_name=f"{exec_name}.{study_name}", 
                gpus=requested_gpus,
                partition=partition if debug is False else 'devel',
                env=env["CONDA_ENV"],
                ld_library_path=env["LD_LIBRARY_PATH"],
                wait=True
            )

        if log_level in ["study", "all"] and debug is not True:
            with open_log(
                os.path.basename(os.path.dirname(os.path.realpath(exec.__file__))).lower(),
                exec_name,
                description=description,
                mode="async",
                sweep_id=f"{exec_name}.{study_name}",
                level_tag="study-level",
                custom_run_id=f"{exec_name}.{study_name}",
            ) as log_study:
                optuna_utils.log_study_metadata(
                    study,
                    log_study,
                    target_names=tuple(config.directions.keys()),
                    log_all_trials=False,
                    log_plot_optimization_history=False,
                    log_plot_edf=False,
                    log_plot_contour=False,
                    log_distributions=False
                )
                import matplotlib.pyplot as plt
                optuna.visualization.matplotlib.plot_contour(study)
                log_study["visualizations/plot_contour"].upload(plt.gcf())
                log_study["config"].upload(config_name)
