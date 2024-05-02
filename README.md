# STUNE: Slurm Tuning Utility
Stune is based on Optuna and simplifies its integration with a SLURM cluster.
In order to use stune on a SLURM cluster, you will need a database server running on a login node that can be accessed by the computed nodes. Stune currently support Redis and PSQL. It is recommended to use Redis as it is easier to install and, accordinly to the Optuna documentation, faster during execution.

## Installation and configuration
We assume you do not have any sudo rights when configuring stune, as this is usally the case when using computing clusters. Thus, we install everything locally and from source. If, instead, you have sudo rights you can probably get aways with a significantly easier installation process. You will still have to go through configuration to allow the chosen database system to work properly with stune. We provide installation tutorials for both Redis and PSQL, but, again, you ONLY NEED ONE. Redis is recommended.

### Redis: installation
Choose your desired installation path. We will be using `~/usr/bin`. Execute the following instructions to install Redis. If any of the steps do not work, please let me know.

```bash
# We follow the tutorial found at https://redis.io/docs/install/install-redis/install-redis-from-source/
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
# this will take a while, you can increase the number of used threads via the flag -j
make -j 16
PREFIX=~/usr/ make install
```
If Redis is installed succesfully we can copy the configuration file to the config folder. We will be using `~/usr/etc/redis`:
```bash
mkdir ~/usr/etc/redis
mv ./redis.conf ~/usr/etc/redis/redis.conf
```
And now we can delete the installation files:
```bash
cd ../
rm -f -r redis-stable*
```
And... done!
Finally, let's create a folder we will use to store temporary stune data, such as the database generated with redis.
```

(We also need to install the pyhton redis package, so make sure to run `pip install "redis[hiredis]"` in your preferred environment)


### Redis: configuration
To use Redis we need to remember the installation and configuration paths. To check that everything works correctly let's open a terminal and run the server:
```bash
# This will tell the server to use the default configuration we copied before
~/usr/bin/redis-server ~/usr/etc/redis/redis.conf
```
If the server does not start and outputs the error
```
# Warning: Could not create server TCP listening socket 127.0.0.1:6379: bind: Address already in use
# Failed listening on port 6379 (tcp), aborting.
```
it means someone is already running something on port 6379 so we need to use a different one:
```bash
~/usr/bin/redis-server ~/usr/etc/redis/redis.conf --port 6380
```
It is also advisable to create a different server storage for each project to keep studies separate. This can be done with the flag `--dir`. In particular, `--dir $(pwd)/.stune` will place the server data
inside the current project's `.stune` configuration folder.

Now that everything is working (hopefully), we can try to connect to the server (remember that if we specified a different port when starting the server we need to it now as well with the `-p` argument).
```bash
# -p is optional
~/usr/bin/redis-cli -p 6380
```
In the redis cli we can now send a `PING` and get a `PONG` back.
```
PING
```

We can now proceed to configure the redis server:
1. **Configure network interface**: the server needs to be discoverable on the local network.
```
CONFIG SET bind 0.0.0.0
```
2. **Enable remote access**: we need to specify a password (here `xxxx`) for the default user in order to log in remotely.
```
CONFIG SET requirepass 'xxxx'
AUTH 'xxxx'
```
We can now save the new configuration and exit:
```
CONFIG REWRITE
exit
```

To close the server use `CTRL+C` in the server's terminal.


### Stune installation and configuration

Clone this repository from GitHub, (select the right branch), and install via pip:
```
pip install -e /path/to/this/repo/ --config-settings editable_mode=strict
```

Stune needs to be configured in each project's root directory you intend to use stune with. Please run `python -m stune.config` in the current project directory to initialize it. To configure `redis` as default storage system, use `redis` as `STUNE_STORAGE`, `''` (empty) as `STUNE_USR`, `xxxx` (your password) as `STUNE_PWD`, and `host_name:host_port` as `STUNE_HOST`, where `host_name` is the address or name of the node running the server (could be `0.0.0.0` if it is on the local machine) and `host_port` is the port the server is listening to (e.g., the default `6379`). These default values are used to compute the default storage address for `stune` in case no storage url is passed via the `--storage` command line argument. The url is computed as `url = $STUNE_STORAGE://$STUNE_USR:$STUNE_PWD@$STUNE_HOST` in case you want to pass it manually to the `--storage` flag.

### Stune tutorial

Once installed, stune can be used as a module by calling it directly in the terminal, using the command

```bash
python -m stune --exe py_file ...
```

You can also list or remove existing optimisation studies using the flags `--ls` and `--rm`. For example,
```bash
python -m stune --exe main --ls
```
will list all the studies whose name start with `main`.

Stune is based on optuna and, as such, it uses mostly the same arguments to configure a study. They can be seen by calling python -m stune -h.

Currently, it has been tested only with direct SSH access. The following is an example of using stune to run an optimization on the file `main.py`:

```bash
python -m stune --exe main --tuner ssh --study pc_analysis --n_tasks 4 --n_trials 128:16 --gpus 0
```

In particular:
- `--tuner ssh` specifies we are using SSH to get direct access of a computing node and we can directly use its resources (cpus and gpus)
- `--study pc_analysis` specifies the suffix of the study. The used study name will be `main.pc_analysis`.
- `--n_tasks 4` specifies we want to run 4 concurrent workers. If `-1` (default) is passed, then as many tasks as possible will be created, according
    to the number of gpus given and the `gpus_per_task` (default 1) argument.
- `--n_trials` specifies how many trials should each worker do. In this case, 128 tasks in groups of 16 (which means that a worker is reset each 16 trials, this seems to be necessary due to existing memory leaks, feel free to play with this number and try with no grouping, i.e., by passing a single number, as well).
- `--gpus 0` specifies we want to use gpu with index 0.
Since no storage url is specified the default values contained in `.stune/config.yaml` are used.

Stune loads three configuration files that contributes to the command line arguments: `.stune/config.yaml`, `main.yaml` (based on the script name), and `main.pc_analysis.yaml` (based on the study name). Optionally, a fourth configuration file can be passed via the `--config` flag. The load order (and thus how values are overwritten) is `./stune/config.yaml > exe > study > custom > cli`.

A suggested project structure would be:
- `py_file.py` is the python script we want to optimise that supports stune (more on this later).
- `py_file.yaml` contains all the configuration necessary to run a study on py_file, such as `gpus_per_task` (which depends on the model/task and thus should be shared among all the studies that uses the same python script as entry point), and the default hyperparameters to run py_file on its own (more on the hyperparameters configuration later).
- `py_file.study.yaml` contains all the specific configuration for a particular study, such as optimization direction (i.e., minimise or maximise the output value), and the hyperparameter space to search over.

#### Using stune in a python script: RunInfo
To make a python script compatible with stune, we simply need to wrap the main code within a function called `main` which accept a single argument of type `RunInfo` and returns the value(s) to optimize:

```python
import stune

def main(config: stune.RunInfo):
    # ...

    return best_accuracy
```

`RunInfo` is a OmegaConf object that contains the loaded configuration values and hyperparemeters (more on this in the next section).
To run the python file outside of stune, simply add the following at the end of it:

```python

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", type=str, help="Configuration file")
    args = parser.parse_args()

    main(stune.RunInfo(stune.load_config(args.get("config"))))
```
and simply use, for example, `python main.py main.yaml`.

#### Study configuration
A configuration file can specify the model hyperaparameters using the `hp` field in a `.yaml` config file. Hyperparameters can be nested:
```yaml
hp:
    optim:
        x:
            lr: 1e-1
        w:
            lr: 1e-4
```
and can be accessed by the RunInfo object using the `/` delimiter (i.e., `x_lr = config['hp/optim/x/lr']).
The keywords `sample_type`, `sample_space` are reserved and cannot be used as hyperparameter names as they are used to specify a hyperparameter space to be optimised.
The following options are available:
- `sample_type: categorical, sample_space: [val1, val2, val3, ...]`
- `sample_type: range, sample_space: [int_from, int_to]` (`int_to` is included)
- `sample_type: float, sample_space: [low, high, step=null, log=False]` (so, for example, `[1e-4, 1e-2, null, True]` or `[0.0, 1.0, 0.1]`)
If a `default` option is provided, it is used when the config file is used outside the optimisation process (such as if I run the `py_file` directly).

#### NOTES:
- there seems to be a bug in Optuna, which prevents the grid sampler to resume after a failed run. As a temporary fix, please comment lines 191 and 219 of `optuna/samplers/_brute_force.py`, removing the entry `TrialState.FAIL` from both dictionaries (the line numbers could change in future versions).