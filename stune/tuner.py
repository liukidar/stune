import os
import math
import subprocess
import random

from .study import Storage, Study
from .slurm import Sbatch
import datetime


class Tuner:
    def __init__(
        self,
        config
    ) -> None:
        self.gpus = config.get("gpus", "").split(",")
        self.n_tasks = config.get("n_tasks", -1)
        self.gpus_per_task = config.get("gpus_per_task", 0)

        self.timeout = config.get("timeout", None)
        try:
            if ":" in self.timeout:
                self.timeout, self.timeout_per_worker = map(float, self.timeout.split(":"))
            else:
                self.timeout = float(self.timeout)
            self.timeout_per_worker = self.timeout
        except TypeError:
            self.timeout_per_worker = self.timeout

        self.n_trials = config.get("n_trials", None)
        try:
            if ":" in self.n_trials:
                self.n_trials, self.n_trials_per_worker = map(int, self.n_trials.split(":"))
            else:
                self.n_trials = int(self.n_trials)
                self.n_trials_per_worker = self.n_trials
        except TypeError:
            self.n_trials_per_worker = self.n_trials

    @staticmethod
    def from_config(config):
        if config.tuner == "ssh":
            tuner = SSHTuner
        elif config.tuner == "slurm":
            tuner = SLURMTuner
        else:
            tuner = Tuner

        return tuner(config)

    def run(self, study: Study, storage: Storage):
        if "gpus" in study.config:
            os.environ["CUDA_VISIBLE_DEVICES"] = study.config["gpus"]
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        _done = study.optimize(storage, n_trials=self.n_trials_per_worker, timeout=self.timeout_per_worker)

        if _done is False:
            if self.n_trials is not None:
                self.n_trials -= self.n_trials_per_worker

                if self.n_trials <= 0:
                    return
            if self.timeout is not None:
                self.timeout -= self.timeout_per_worker

                if self.timeout <= 0:
                    return
            self.schedule(study, storage, n=1)

    def schedule(self, study: Study, storage: Storage):
        raise NotImplementedError


class SSHTuner(Tuner):
    def schedule(self, study: Study, storage: Storage, n: int | None = None):
        if n is None:
            n = self.n_tasks

        if n == -1:
            n = int(len(self.gpus) / self.gpus_per_task) if self.gpus_per_task else 1

        if self.timeout is not None:
            cmd_timeout = f"--timeout {self.timeout}:{self.timeout_per_worker}"
        else:
            cmd_timeout = ""

        if self.n_trials is not None:
            cmd_trials = f"--n_trials {self.n_trials}:{self.n_trials_per_worker}"
        else:
            cmd_trials = ""

        processes = []
        for i in range(n):
            gpus = tuple(map(
                lambda x: self.gpus[x % len(self.gpus)],
                range(math.floor(self.gpus_per_task * i), math.ceil(self.gpus_per_task * (i + 1)))
            ))
            cmd_gpus = f"--gpus={','.join(gpus)}" if len(gpus) else ""
            current_datetime = datetime.datetime.now()
            datetime_string = current_datetime.strftime("%m%d:%H%M")
            cmd = (
                f"python -m stune \
                --from_config '.stune/configs/{study.name}.cfg' \
                {cmd_gpus} {cmd_timeout} {cmd_trials} \
                > .stune/output/{study.id(storage)}-{study.name}-{datetime_string}-{random.randint(0, 9999)}.out 2>&1"
            )
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)

        for process in processes:
            process.wait()


class SLURMTuner(Tuner):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.partition = config["partition"]
        self.time = config["time"]
        self.conda_env = config["CONDA_ENV"]
        self.n_jobs = config.get("n_jobs", 1)
        self.cpus_per_task = config.get("cpus_per_task", 1)

    def schedule(self, study: Study, storage: Storage, n: int | None = None):
        if int(os.environ.get("SLURM_PROCID", 0)) != int(os.environ.get("SLURM_NTASKS", 1)) - 1:
            return

        tasks_per_job = self.n_tasks
        gpus_per_job = math.ceil(self.gpus_per_task)
        if tasks_per_job == -1:
            tasks_per_job = int(gpus_per_job / self.gpus_per_task) if self.gpus_per_task else 1

        cmd = (
            f"python -m stune \
            --from_config '.stune/configs/{study.name}.cfg'"
        )

        sbatch = Sbatch(
            cmd,
            tasks_per_job,
            self.cpus_per_task,
            self.time,
            study.name,
            gpus_per_job,
            self.partition,
            output=f".stune/output/{study.id(storage)}-%j-%x.out",
            env=self.conda_env
        )

        sbatch.submit(n or self.n_jobs)
