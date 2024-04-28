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

Stune needs to be configured in each project's root directory you intend to use stune with. Please run `python -m stune.config` in the current project directory to initialize it. To configure `redis` as default storage system, use `redis` as `STUNE_STORAGE`, ` ` as `STUNE_USR`, `xxxx` as `STUNE_PWD`, and `host_name:host_port` as `STUNE_HOST`, where `host_name` is the address or name of the node running the server (could be `0.0.0.0` if it is on the local machine) and `host_port` is the port the server is listening to (e.g., the default `6379 `). These default values are used to compute the default storage address for `stune` in case no storage url is passed via the `--storage` command line argument. The url is computed as `url = $STUNE_STORAGE://$STUNE_USR:$STUNE_PWD@$STUNE_HOST`.