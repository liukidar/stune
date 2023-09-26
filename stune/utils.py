from typing import Any, Optional, List
import re

import omegaconf
from omegaconf import OmegaConf


class RunInfo:
    def __init__(self,
        study_name: str,
        config: OmegaConf,
        log = None,
        trial = None
    ) -> None:
        self.study_name = study_name
        self.config = config
        self.log = log or {}
        self.trial = trial
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
