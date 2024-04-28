from typing import Any

import optuna
import datetime
import functools
import importlib
from omegaconf import OmegaConf

from .utils import RunInfo


class Storage:
    def __init__(self, url) -> None:
        self.url = url
        self.storage = None

    @staticmethod
    def from_config(config: OmegaConf):
        if "storage" in config:
            url = config["storage"]
        else:
            if "STUNE_STORAGE" not in config:
                url = None
            else:
                url = f"{config['STUNE_STORAGE']}://{config['STUNE_USR']}:{config['STUNE_PWD']}@{config['STUNE_HOST']}"

        return Storage(url)

    def get(self):
        if self.storage is None:
            self.storage = self._make_storage()

        return self.storage

    def clear_stale_trials(self, study_name, timeeout_minutes=60):
        try:
            study = optuna.study.load_study(study_name=study_name, storage=self.get())
            active_runs = study.get_trials(
                deepcopy=False, states=[optuna.trial.TrialState.RUNNING]
            )

            time_now = datetime.datetime.now()

            for run in active_runs:
                if (time_now - run.datetime_start).seconds > timeeout_minutes * 60:
                    study._storage.set_trial_state_values(
                        run._trial_id, optuna.trial.TrialState.WAITING
                    )
        except KeyError:
            pass

    def _make_storage(self):
        if self.url is None:
            return optuna.storages.InMemoryStorage()
        elif self.url.startswith("redis://"):
            return optuna.storages.JournalStorage(
                optuna.storages.JournalRedisStorage(url=self.url)
            )
        elif self.url.startswith("postgresql://"):
            return optuna.storages.RDBStorage(
                url=self.url, heartbeat_interval=60, grace_period=120
            )
        else:
            raise NotImplementedError(f"Storage {self.url} is not supported")


class CountExecutedTrialsCallback:
    def __init__(self) -> None:
        self.n_trials_failed = 0
        self.n_trials_completed = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]:
            self.n_trials_completed += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.n_trials_failed += 1


class Study:
    def __init__(
        self,
        config: OmegaConf,
    ):
        self._study = None
        self.config = config

    @staticmethod
    def from_config(config: OmegaConf):
        return Study(
            config
        )

    @property
    def name(self):
        return self.config["study"]

    def id(self, storage: Storage):
        return self.get(storage)._study_id

    def get(self, storage: Storage) -> optuna.Study:
        if self._study is None:
            self._study = self._make_study(storage)

        return self._study

    def optimize(self, storage: Storage) -> bool:
        def worker(
            trial: Any,
            study: optuna.Study,
            exe: Any,
            config: OmegaConf
        ):
            _run_info = RunInfo(config, study.study_name, trial)

            return exe.main(_run_info)

        _study = self.get(storage)
        _counterCallback = CountExecutedTrialsCallback()
        _exe = importlib.import_module(self.config["exe"])

        _study.optimize(
            functools.partial(
                worker,
                study=self._study,
                exe=_exe,
                config=self.config
            ),
            n_trials=self.config.get("n_trials", 1),
            n_jobs=1,  # we use process parallelism and not thread parallelism
            callbacks=[_counterCallback],
            gc_after_trial=self.config.get("gc_after_trial", True)
        )
        storage.clear_stale_trials(_study.study_name)

        # Return if the study is completed
        return not (
            _counterCallback.n_trials_completed == self.config.get("n_trials", 1)
            or _counterCallback.n_trials_failed != 0
        )

    def _make_study(self, storage: Storage):
        samplers = {
            None: lambda: None,
            "random": optuna.samplers.RandomSampler,
            "grid": optuna.samplers.BruteForceSampler,
        }
        _sampler = self.config.get("sampler", None)
        if _sampler not in samplers:
            raise NotImplementedError(f"Sampler {_sampler} is not supported")

        return optuna.create_study(
            study_name=self.name,
            storage=storage.get(),
            load_if_exists=self.config.get("load_if_exists", True),
            sampler=samplers[_sampler](),
        )
