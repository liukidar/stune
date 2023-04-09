from typing import Any
import re

import omegaconf
from omegaconf import OmegaConf


def _sample_params(source: OmegaConf, target: Any, root: str = "", trial = None):
    for param_name in source:
        param_value = source[param_name]

        if isinstance(param_value, omegaconf.DictConfig):
            sample_from = param_value.get("sample_from", None)
            
            if sample_from is not None:
                if len(sample_from) == 3:
                    suggest_type, suggest_values, suggest_params = sample_from
                elif len(sample_from) == 2:
                    suggest_type, suggest_values = sample_from
                    suggest_params = {}
                else:
                    raise ValueError(f"sample from should be a list of either 2 or 3 elements.")

                suggest_type = getattr(trial, f"suggest_{suggest_type}")
                suggest_params = dict(suggest_params)

                target[root + param_name] = suggest_type(root + param_name, *suggest_values, **suggest_params)

                continue

            value = param_value.get("value", None)
            if value is not None:
                target[root + param_name] = value
                
                continue

            _sample_params(param_value, target, root + param_name + "/", trial)
        else:
            target[root + param_name] = param_value


class RunInfo:
    def __init__(self,
        params: OmegaConf,
        log = None,
        trial = None
    ) -> None:
        sampled_params = {}
        OmegaConf.register_new_resolver("ev", lambda py_code: eval(re.sub(r"([^\W0-9][\w/]*)", r"sampled_params['\1']", py_code), {"sampled_params": sampled_params}))
        _sample_params(params, sampled_params, "", trial)
        OmegaConf.clear_resolver("ev")

        if log is not None:
            log['params'] = sampled_params

        self.trial = trial
        self.log = log
        self.params = sampled_params

    def __getitem__(self, i: Any) -> Any:
        return self.params[i]

    def __setitem__(self, i: Any, v: Any) -> None:
        self.params[i] = v

        if self.log is not None:
            self.log['params/' + i] = v

    def is_log(self):
        return self.log is not None

    def is_master(self):
        return self.trial is None
