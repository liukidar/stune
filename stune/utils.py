from typing import Any, Optional, List

import optuna
import omegaconf
from omegaconf import OmegaConf


def load_config(config: str | None = None):
    if config is None:
        return OmegaConf.create()

    extension = ".yaml"
    if not str(config).endswith(extension):
        config += extension

    # Load exec configuration
    try:
        config = OmegaConf.load(config)
    except FileNotFoundError:
        config = OmegaConf.create()

    return config


class RunInfo:
    def __init__(
        self,
        config: OmegaConf,
        study_name: str = None,
        trial: Optional[optuna.Trial] = None,
        log=None,
    ) -> None:
        self.config = config
        self.study_name = study_name
        self.trial = trial
        self.log = log or {}
        self.locked = False

        OmegaConf.register_new_resolver(
            "py", lambda code: eval(code.strip()), replace=True
        )
        OmegaConf.register_new_resolver(
            "hp", lambda param: self[f"hp/{param}"], replace=True
        )

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

    def _sample_param(self, key, param, trial: optuna.Trial):
        if trial is None:
            return param.get("default", None)

        sample_type = param.get("sample_type", None)

        if sample_type is not None:
            if sample_type == "single_value":
                param = param["sample_space"]
            elif sample_type == "categorical":
                param = trial.suggest_categorical(key, param["sample_space"])
            elif sample_type == "float":
                param = trial.suggest_float(
                    key,
                    *param["sample_space"][0],
                    step=param["sample_space"][1] if len(param["sample_space"]) > 1 else None,
                    log=param["sample_space"][2] if len(param["sample_space"]) > 2 else False,
                )
            elif sample_type == "range":
                param = trial.suggest_categorical(
                    key,
                    tuple(
                        range(param["sample_space"][0], param["sample_space"][1] + 1)
                    ),
                )
            else:
                raise NotImplementedError(f"sample_type {sample_type} is not supported")
        else:
            return NotImplementedError(
                "At the moment it is not possible to return dictionaries when querying parameters."
            )

        return param
