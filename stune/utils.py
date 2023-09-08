from typing import Any
import re

import omegaconf
from omegaconf import OmegaConf


class RunInfo:
    def __init__(self,
        name: str,
        params: OmegaConf,
        log = None,
        trial = None
    ) -> None:
        self.name = name
        self.params = params
        self.log = log or {}
        self.trial = trial
        
        OmegaConf.register_new_resolver("py", lambda code: eval(code.strip()), replace=True)
        OmegaConf.register_new_resolver("hp", lambda param: self[f"hp/{param}"], replace=True)

    def __getitem__(self, i: Any) -> Any:
        if i in self.log:
            return self.log[i]

        path = i.split("/")
        param = self.params
        for key in path:
            param = param[key]

        if isinstance(param, omegaconf.DictConfig):
            param = self._sample_param(i, param, self.trial)
        
        self.log[i] = param

        return param

    def __setitem__(self, i: Any, v: Any) -> None:
        path = i.split("/")
        param = self.params
        for key in path[:-1]:
            param = param[key]
        param[path[-1]] = v

        self.log[i] = v      

    def is_log(self):
        return self.log is not None

    def is_master(self):
        return self.trial is None
    
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
        
        return param
