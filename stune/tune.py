import os
from typing import Any, Optional
import importlib
from datetime import datetime
import functools
import time

from omegaconf import OmegaConf
import optuna

from .utils import Study, Storage
from .slurm import Sbatch
from . import RunInfo #, open_log


def print_time(str: Optional[str] = None) -> None:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', str)


def worker(
    trial: Any,
    study: Study,
    exec: Any,
    params: OmegaConf,
    log_mode: Optional[str] = None
):
    if log_mode is not None:
        project_name = os.path.basename(os.path.dirname(os.path.realpath(exec.__file__))).lower()
        # with open_log(
        #     project_name,
        #     study.exec_name,
        #     mode=log_mode,
        #     sweep_id=study.name,
        #     level_tag="trial-level"
        # ) as log:
        #     run_info = RunInfo(params, study.name, trial, log)
    else:
        run_info = RunInfo(params, study.name, trial, None)

    return exec.main(run_info)


def run(
    env,
    study: Study,
    storage: Storage,
    config_name: str,
    debug: bool = False,
    log_level: Optional[str] = None,
    description: Optional[str] = None
):
    parallelisation_mode = "process"
    config = OmegaConf.load(config_name)
    gpus_per_task = config.gpus_per_task if debug is False else 1
    tasks_per_node = max(1, int(1 / gpus_per_task))

    if parallelisation_mode == "thread":
        # Paarallelisation is handled by the worker
        # so we reserve all cpus at once
        cpus_per_task = config.cpus_per_task * tasks_per_node
        n_processes = 1
        jobs_per_process = tasks_per_node
    elif parallelisation_mode == "process":
        # Parallelisation is handled by the scheduler
        cpus_per_task = config.cpus_per_task
        n_processes = tasks_per_node
        jobs_per_process = 1

    # Worker
    if study.is_worker() or study.n_jobs == 0:
        if log_level in ["trial", "all"]:
            log_mode = "debug" if debug is True else "offline"
        else:
            log_mode = None
        
        exec = importlib.import_module(study.exec_name)
        study.get(storage).optimize(
            functools.partial(
                worker,
                study=study,
                exec=exec,
                params=config,
                log_mode=log_mode,
            ),
            n_trials=study.trials_per_worker or max(1, study.n_trials // (study.n_jobs * tasks_per_node)),
            n_jobs=jobs_per_process,
            gc_after_trial=True
        )
    # Scheduler
    else:
        cmd = f"python -m stune {study.exec_name} "
        cmd += study.cmd_str()
        cmd += storage.cmd_str()
        if debug is True:
            cmd += " --debug" 

        time_minutes = (60 * 6) if study.partition != "devel" else 60
        requested_gpus = max(1, gpus_per_task)
        trials_executed = -1
        trials_in_study = len(study.get(storage).get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        
        sbatch = Sbatch(
            cmd,
            tasks_per_node=n_processes,
            cpus_per_task=cpus_per_task,
            gpu_reserved_memory=float(env["GPU_MEM_RESERVED"]),
            time_minutes=time_minutes,
            job_name=study.name, 
            gpus=requested_gpus,
            partition=study.partition if debug is False else "devel",
            env=env["CONDA_ENV"],
            ld_library_path=env["LD_LIBRARY_PATH"],
            resources=config.get("resources", None)
        )
        jobs = []

        # We run on the assumptions that a worker will finish before the timeout
        # so that there a no running trials to clean up since we cannot distinguish
        # them from the ones that are still running.
        try:
            while (
                (
                    study.n_trials > 0 
                    and trials_in_study < study.n_trials
                )
                or study.n_trials == 0
            ) and trials_in_study > trials_executed:
                trials_executed = trials_in_study
                while len(jobs) < study.n_jobs:
                    jobs.append(sbatch.submit())

                # Wait for a job to finish
                while True:
                    for job in jobs:
                        if job.poll() is not None:
                            jobs.remove(job)
                            break
                    
                    if len(jobs) < study.n_jobs:
                        break

                    time.sleep(60)
                trials_in_study = len(study.get(storage).get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        finally:
            exec = importlib.import_module(study.exec_name)
            if hasattr(exec, "cleanup"):
                exec.cleanup(study.name)
            storage.clear_running_trials(study.name)



        # if log_level in ["study", "all"] and debug is not True:
        #     import neptune.integrations.optuna as optuna_utils
        #     import optuna.visualization
        #     import matplotlib.pyplot as plt

        #     with open_log(
        #         os.path.basename(os.path.dirname(os.path.realpath(exec.__file__))).lower(),
        #         study.exec_name,
        #         description=description,
        #         mode="async",
        #         sweep_id=study.name,
        #         level_tag="study-level",
        #         custom_run_id=study.name,
        #     ) as log_study:
        #         optuna_utils.log_study_metadata(
        #             study,
        #             log_study,
        #             target_names=tuple(config.directions.keys()),
        #             log_all_trials=False,
        #             log_plot_optimization_history=False,
        #             log_plot_edf=False,
        #             log_plot_contour=False,
        #             log_distributions=False
        #         )
        #         optuna.visualization.matplotlib.plot_contour(study)
        #         log_study["visualizations/plot_contour"].upload(plt.gcf())
        #         log_study["config"].upload(config_name)
