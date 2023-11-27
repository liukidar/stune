import os
from typing import Any, Optional
import importlib
import functools
import subprocess
import datetime

from omegaconf import OmegaConf
import optuna

from .utils import Study, Storage
from .slurm import Sbatch
from . import RunInfo #, open_log


class TimeoutCallback:
    def __init__(self, reserved_minutes: int|float) -> None:
        self.timeout = reserved_minutes * 60
        self.time_per_trial = 0
        self.start_time = datetime.datetime.now()
        self.last_trial_time = self.start_time
        self.timed_out = False
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        time_now = datetime.datetime.now()
        last_trial_time = (time_now - self.last_trial_time).seconds

        # If the trial took longer than the expected time, update the expected time
        if last_trial_time > self.time_per_trial:
            self.time_per_trial = last_trial_time

        # Check if running another trial would exceed the timeout (with a margin of 2 trials)
        if (datetime.datetime.now() - self.start_time).seconds + self.time_per_trial * 2 > self.timeout:
            study.stop()
            self.timed_out = True
        
        self.last_trial_time = time_now


class CountExecutedTrialsCallback:
    def __init__(self) -> None:
        self.n_trials_failed = 0
        self.n_trials_completed = 0
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]:
            self.n_trials_completed += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.n_trials_failed += 1


# Returns the maximum time in minutes that can be requested for a given partition
def query_partition_maxtime(partition: str) -> int:
    cmd = f"sinfo -p {partition} -o %l"
    result = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)
    max_time = result.stdout.decode("utf-8").split("\n")[1]

    hours, minutes, _ = max_time.split(":")

    if "-" in hours:
        days, hours = hours.split("-")
        hours = int(days) * 24 + int(hours)
    else:
        hours = int(hours)
    
    max_minutes = int(minutes) + hours * 60

    return max_minutes


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

    # Read config
    gpus_per_task = config.get("gpus_per_task", 1) if debug is False else 1
    tasks_per_node = max(1, int(1 / gpus_per_task))
    trials_per_worker = study.trials_per_worker # TODO: or max(1, study.n_trials // (study.n_jobs * tasks_per_node))
    minutes_per_trials = config.get("minutes_per_trial", 60)
    reserved_minutes = int(min(minutes_per_trials * (trials_per_worker + 1) * 1.2, query_partition_maxtime(study.partition) - 1))

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
    
    # Define job to be (re-)submitted
    cmd = f"python -m stune {study.exec_name} "
    cmd += study.cmd_str()
    cmd += storage.cmd_str()
    if debug is True:
        cmd += " --debug"
    
    sbatch = Sbatch(
        cmd,
        tasks_per_node=n_processes,
        cpus_per_task=cpus_per_task,
        gpu_reserved_memory=float(env["GPU_MEM_RESERVED"]),
        time_minutes=reserved_minutes,
        job_name=study.name, 
        gpus=max(1, gpus_per_task),
        partition=study.partition,
        env=env["CONDA_ENV"],
        ld_library_path=env["LD_LIBRARY_PATH"],
        resources=config.get("resources", None)
    )

    # Worker
    if study.is_worker() or study.n_jobs == 0:
        if log_level in ["trial", "all"]:
            log_mode = "debug" if debug is True else "offline"
        else:
            log_mode = None
                
        counter_callback = CountExecutedTrialsCallback()
        timeout_callback = TimeoutCallback(reserved_minutes)
        exec = importlib.import_module(study.exec_name)
        study.get(storage).optimize(
            functools.partial(
                worker,
                study=study,
                exec=exec,
                params=config,
                log_mode=log_mode,
            ),
            n_trials=trials_per_worker,
            n_jobs=jobs_per_process,
            callbacks=[counter_callback, timeout_callback],
            gc_after_trial=True
        )
        storage.clear_stale_trials(study.name)

        # Once done check if study is complete,
        # if not, schedule another worker
        if (
            study.is_worker()
            and int(os.environ["SLURM_PROCID"]) == int(os.environ["SLURM_NTASKS"]) - 1
            and (
                (
                    # TODO: redefine this behaviour
                    study.n_trials > 0
                )
                or study.n_trials == 0
            )
            and (
                counter_callback.n_trials_completed == trials_per_worker
                or counter_callback.n_trials_failed != 0
                or timeout_callback.timed_out is True
            )
        ):
            sbatch.submit()

    # Scheduler
    else:
        storage.clear_stale_trials(study.name)
        sbatch.submit(study.n_jobs)

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
