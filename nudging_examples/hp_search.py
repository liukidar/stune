from typing import Callable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

# Core dependencies
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from omegacli import OmegaConf

# stune
import stune
import json

import random

from models import get_model
from datasets import get_loader

def get_datasetinfo(dataset):
    if dataset == "MNIST":
        return 10, 28
    elif dataset == "FashionMNIST":
        return 10, 28
    elif dataset == "CIFAR10":
        return 10, 32
    elif dataset == "CIFAR100":
        return 100, 32
    elif dataset == "TinyImageNet":
        return 200, 56
    else:
        raise ValueError("Invalid dataset name")

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model , beta=1.0):
    return model(x, y, beta=beta)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model ):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model , optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model, beta=beta)
    optim_h.init(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True))(model))


    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"], True)
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache | pxc.VodeParam):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], mul=1/beta)
    
    
@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model ):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache | pxc.VodeParam):
        y_ = forward(x, None, model=model).argmax(axis=-1)


    return (y_ == y).mean(), y_


def train(dl, T, *, model , optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w, optim_h=optim_h, beta=beta
        )


def eval(dl, *, model ):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def main(run_info: stune.RunInfo):

    assert run_info["study"] is not None, "Study name must be provided"
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    intermidiate_savepath = os.path.join(results_dir, run_info["study"] + '.json')
   
    dataset_name = run_info["study"].split('_')[-1]

    if run_info["hp/beta_factor"] == 0:
        run_info["hp/Nudging_Mode"] = "CN"
    elif run_info["hp/beta_factor"] == -1:
        run_info["hp/Nudging_Mode"] = "NN"
    elif run_info["hp/beta_factor"] == 1:
        if run_info["hp/beta"] == 1:
                if run_info["hp/se_flag"]:
                    run_info["hp/Nudging_Mode"] = "SE"
                else:
                    run_info["hp/Nudging_Mode"] = "CE"
        else:
            run_info["hp/Nudging_Mode"] = "PN"

    run_info["hp/dataset"] = dataset_name

    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    nm_classes, input_size = get_datasetinfo(dataset_name)

    model = get_model(
        model_name=run_info["hp/model"], 
        nm_classes=nm_classes, 
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]),
        input_size=input_size,
        se_flag=run_info["hp/se_flag"])
    
    train_dataloader, test_dataloader = get_loader(dataset_name, batch_size)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],
        peak_value=1.1 * run_info["hp/optim/w/lr"],
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,
        decay_steps=len(train_dataloader)*nm_epochs,
        end_value=0.1 * run_info["hp/optim/w/lr"],
        exponent=1.0)

    optim_h = pxu.Optim(
        optax.chain(
            optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"]),
        )
    )
    optim_w = pxu.Optim(optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]), pxu.Mask(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    acc_list = []
    below_times = 0
    stop_increase = 0
    
    for e in range(nm_epochs):
        if run_info["hp/beta"] == 1 and run_info["hp/beta_ir"] != 0:
            acc_list.append(0)
            break
        if run_info["hp/beta_factor"] == 0:
            beta = random.choice([-1,1]) * (run_info["hp/beta"] + run_info["hp/beta_ir"]*e)
        else:
            beta = run_info["hp/beta_factor"] * (run_info["hp/beta"] + run_info["hp/beta_ir"]*e)

        if beta >= 1.0:
            beta = 1.0
        elif beta <= -1.0:
            beta = -1.0

        train(train_dataloader, T=run_info["hp/T"], model=model, optim_w=optim_w, optim_h=optim_h, beta=beta)
        a, y = eval(test_dataloader, model=model)
        acc_list.append([float(a), beta])
        if run_info.study_name is None:
            print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
        if a > best_accuracy:
            best_accuracy = a
            stop_increase = 0
        else:
            stop_increase += 1
        if e > 15 and float(a) < 0.2:
            below_times += 1
        else:
            below_times = 0
        if below_times > 5 or (e >= 25 and stop_increase > 5):
            break
    
    config_save = run_info.log
    config_save['results'] = acc_list
    try:
        with open(intermidiate_savepath, 'r') as file:
            # load json file if exists
            data = json.load(file)
    except FileNotFoundError:
        data = []
    except json.decoder.JSONDecodeError:
        data = []

    # add new data to the list
    data.append(config_save)

    with open(intermidiate_savepath, 'w') as file:
        json.dump(data, file, indent=4)

    return best_accuracy


if __name__ == "__main__":
    import sys
    run_info = stune.RunInfo(
        OmegaConf.load(sys.argv[1])
    )
    main(run_info)
