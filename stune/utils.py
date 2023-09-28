from typing import Any, Optional, List
from pathlib import Path

import omegaconf
from omegaconf import OmegaConf


def get_storage(args, env):
    storage = args.storage if args.storage else None
    if args.debug is False and storage is None:
        storage = f"postgresql://{env['PSQL_USR']}:{env['PSQL_PWD']}@{env['PSQL_HOST']}"

    return storage


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
