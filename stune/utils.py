from typing import Any, Optional, List
from pathlib import Path

import optuna
import omegaconf
from omegaconf import OmegaConf


class Storage:
    def __init__(self, url) -> None:
        self.url = url
        self.storage = None
    
    @staticmethod
    def init(args, env):
        url = args.storage
        if args.debug:
            url = None
        else:
            url = args.storage
            if url is None:
                url = f"{env['STUNE_STORAGE']}://{env['STUNE_USR']}:{env['STUNE_PWD']}@{env['STUNE_HOST']}"

        return Storage(url)
    
    def get(self):
        if self.storage is None:
            self.storage = self._make_storage()
        
        return self.storage
    
    def cmd_str(self):
        return f" --storage {self.url} "
    
    def clear_running_trials(self, study_name):
        study = optuna.study.load_study(study_name=study_name, storage=self.get())
        zombie_runs = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.RUNNING])

        for run in zombie_runs:
            study._storage.set_trial_state_values(run._trial_id, optuna.trial.TrialState.WAITING)
    
    def _make_storage(self):
        if self.url is None:
            return optuna.storages.InMemoryStorage()
        elif self.url.startswith("redis://"):
            return optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(url=self.url))
        elif self.url.startswith("postgresql://"):
            return optuna.storages.RDBStorage(url=self.url, heartbeat_interval=60, grace_period=120)
        else:
            raise NotImplementedError(f"Storage {self.url} is not supported")


def load_config(
        exec: str,
        study: Optional[str] = None,
        config: Optional[str] = None
    ):
    def ensure_extension(path, extension, only_suffix: bool = True):
        if not str(path).endswith(extension):
            if only_suffix is True:
                return Path(path).with_suffix(extension)
            else:
                return Path(str(path) + extension)
        else:
            return path

    # Load exec configuration
    try:
        exec_config = OmegaConf.load(ensure_extension(exec, ".yaml"))
    except FileNotFoundError:
        exec_config = OmegaConf.create()

    # Load study configuration
    if study:
        try:
            study_config = OmegaConf.load(ensure_extension(study, ".yaml", False))
        except FileNotFoundError:
            study_config = OmegaConf.create()
    else:
        study = study_config = OmegaConf.create()

    # Load manual configuration
    if config:
        manual_config = OmegaConf.load(ensure_extension(config, ".yaml", False))
    else:
        manual_config = OmegaConf.create()

    config = OmegaConf.unsafe_merge(
        exec_config,
        study_config,
        manual_config
    )

    return config


class Study:
    _study: optuna.Study = None

    def __init__(
        self,
        exec_name: str,
        study_name: str,
        sampler: str,
        partition: Optional[str] = None,
        n_jobs: Optional[int] = None,
        n_trials: int = 1,
        trials_per_worker: Optional[int] = None,
        load_if_exists: bool = True,
    ):
        self.exec_name = exec_name
        self.study_name = study_name
        self.sampler = sampler
        self.partition = partition
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.trials_per_worker = trials_per_worker
        self.load_if_exists = load_if_exists

    @staticmethod
    def init(args, exec_name: str, study_name: Optional[str] = None, load_if_exists: bool = True):
        if ":" in args.n_trials:
            n_trials, trials_per_worker = args.n_trials.split(":")
        else:
            n_trials = args.n_trials
            trials_per_worker = None

        return Study(
            exec_name=exec_name,
            study_name=study_name or args.study,
            sampler=args.sampler,
            partition=args.partition,
            n_jobs=args.n_jobs,
            n_trials=int(n_trials),
            trials_per_worker=int(trials_per_worker),
            load_if_exists=load_if_exists
        )
    
    def get(self, storage: Storage) -> optuna.Study:
        if self._study is None:
            self._study = self._make_study(storage)
        
        return self._study
    
    @property
    def name(self):
        return f"{self.exec_name}.{self.study_name}"
    
    def cmd_str(self):
        cmd = f" --study {self.study_name} "
        if self.sampler is not None:
            cmd += f"--sampler {self.sampler} "
        if self.partition is not None:
            cmd += f"--partition {self.partition} "
        if self.n_jobs is not None:
            cmd += f"--n_jobs -1 "
        if self.n_trials is not None:
            cmd += f"--n_trials {self.n_trials}:{self.trials_per_worker} " # TODO
        
        return cmd

    def is_worker(self):
        return self.n_jobs == -1

    def _make_study(self, storage: Storage):
        samplers = {
            None: lambda: None,
            "random": optuna.samplers.RandomSampler,
            "grid": optuna.samplers.BruteForceSampler
        }

        if self.sampler not in samplers:
            raise NotImplementedError(f"Sampler {self.sampler} is not supported")
        sampler = samplers[self.sampler]()

        return optuna.create_study(
            study_name=self.name,
            storage=storage.get(),
            load_if_exists=self.load_if_exists,
            sampler=sampler
        )


class RunInfo:
    def __init__(self,
        config: OmegaConf,
        study_name: str = None,
        trial = None,
        log = None
    ) -> None:
        self.config = config
        self.study_name = study_name
        self.trial = trial
        self.log = log or {}
        self.locked = False
        
        OmegaConf.register_new_resolver("py", lambda code: eval(code.strip()), replace=True)
        OmegaConf.register_new_resolver("hp", lambda param: self[f"hp/{param}"], replace=True)

    def __getitem__(self, i: Any) -> Any:
        if i in self.log:
            return self.log[i]
        
        if self.locked is True:
            raise PermissionError("Cannot access new parameter from a locked RunInfo")

        path = i.split("/")
        param = self.config
        for key in path:
            param = param[key]

        if isinstance(param, omegaconf.DictConfig):
            param = self._sample_param(i, param, self.trial)
        
        self.log[i] = param

        return param

    def __setitem__(self, i: Any, v: Any) -> None:
        if self.locked is True:
            raise PermissionError("Cannot modify parameter in a locked RunInfo")

        path = i.split("/")
        param = self.config
        for key in path[:-1]:
            param = param[key]
        param[path[-1]] = v

        self.log[i] = v
    
    def lock(self, to_load: List[str] = []):
        # Load required elements before locking
        for p in to_load:
            self.__getitem__(p)

        self.locked = True

    @property
    def trial_id(self):
        return self.trial.number if self.trial is not None else None
    
    def _sample_param(self, key, param, trial):
        if trial is None:
            return param.get("default", None)

        sample_type = param.get("sample_type", None)

        if sample_type is not None:
            if sample_type == "single_value":
                param = param["sample_space"]
            elif sample_type == "categorical":
                param = trial.suggest_categorical(key, param["sample_space"])
            elif sample_type == "float":
                param = trial.suggest_float(key, *param["sample_space"])
            elif sample_type == "range":
                param = trial.suggest_categorical(key,
                                                  tuple(range(param["sample_space"][0], param["sample_space"][1] + 1)))
            else:
                raise NotImplementedError(f"sample_type {sample_type} is not supported")
        else:
            return NotImplementedError("At the moment it is not possible to return dictionaries when querying parameters.")
        
        return param
