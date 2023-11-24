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

### Installation and configuration: PSQL
In order to use stune, you will need a working postgreSQL server accessible by compute nodes.

#### LOCAL USER
It is possible to install it without any sudo permissions.
You will need to execute the following commands (on JADE 2 make sure to be on a login node to have internet access via wget). If any of the following steps do not work, please let me know.

```bash
# Change XX.Y with whichever version you intend to use. You can visit https://ftp.postgresql.org/pub/source/ to find the most recent one (currently 16.0).
wget https://ftp.postgresql.org/pub/source/vXX.Y/postgresql-XX.Y.tar.bz2
tar xf postgresql-XX.Y.tar.bz2
cd postgresql-XX.Y

# We recommend to install psql into your conda environment and, consequently, use `$CONDA_PREFIX/psql` for `prefix` and `$CONDA_PREFIX/bin/python` for `with-python PYTHON=`:
# ./configure --prefix=$CONDA_PREFIX/psql --with-python PYTHON=$CONDA_PREFIX/bin/python --without-readline --without-icu
# NOTE: on JADE 2 the following make command will fail unless you also set `--without-readline` and ` --without-icu` so it is specified by default. It could be that other systems have, instead, the required libraries to avoid those exclusions.
./configure --prefix=target/psql/installation/path/ --with-python PYTHON=path/to/python/installation --without-readline --without-icu

# 8 is the number of cores, use more if you have!
make -j 8
make install
```
Once installed, to intialize the server run
```bash
# DEFAULT:
# $CONDA_PREFIX/psql/bin/initdb -D $CONDA_PREFIX/psql/data
target/psql/installation/path/bin/initdb -D target/psql/installation/path/data
```

If all is good you can now remove your installation data:
```bash
rm -f -r postgresql-*
```

Let's start the server and see if we can connect to it locally.

```bash
# DEFAULT:
# $CONDA_PREFIX/psql/bin/pg_ctl -D $CONDA_PREFIX/psql/data start
target/psql/installation/path/bin/pg_ctl -D target/psql/installation/path/data start

```

We can now connect to the newly created database:

```bash
# DEFAULT:
# $CONDA_PREFIX/psql/bin/psql -d postgres
target/psql/installation/path/bin/psql -d postgres
```

Create our own (within the psql shell):

```bash
# your-username is the user name you use to login to the cluster.
# For example, mine on JADE looks like qjt12-xxz04
CREATE DATABASE "your-username";

# Connect to the new database;
\c "your-username"

# Set a password for the your-username user (note the different quotes)
ALTER ROLE current_role WITH password 'xxxxxxxx';

# Done. We can exit now.
\q
```

Unfortunately there is still quite a lot to do as, by default, PostgreSQL does not accept external connections. To change it, we need to edit the file `target/psql/installation/path/data/pg_hba.conf` (default: `$CONDA_PREFIX/psql/data/pg_hba.conf`) and add at the end of it, under "# IPv4 local connections:", the line:
```bash
# Mine looks like:
# host    qjt12-xxz04   qjt12-xxz04   all    password
host    your-username   your-username   all    password
```

We also need to edit the file `target/psql/installation/path/data/postgresql.conf` (default `$CONDA_PREFIX/psql/data/postgresql.conf`) by decommenting the line `listen_addresses = 'localhost'` and changing it to `listen_addresses = '*'` (optional: I also suggest to change `max_connections` to something like 512 and `shared_buffers` to 512MB).

Save everything and restart the psql server by running
```bash
# DEFAULT:
# CONDA_PREFIX/psql/bin/pg_ctl -D $CONDA_PREFIX/psql/data restart
target/psql/installation/path/bin/pg_ctl -D target/psql/installation/path/data restart
```

Finally, if all went correctly, we should be able to connect to our psql server:
```bash
# DEFAULT:
# $CONDA_PREFIX/psql/bin/psql -U qjt12-xxz04 -h dgk227
target/psql/installation/path/bin/psql -U your-username -h hostname
```
NOTE: the server was started from login node `dgk227`, which thus becomes is the database hostname. I suggest starting psql from a screen so that you can keep it alive and always find which node is using (remember that if you followed the default configuration, PostreSQL is bound to to your CONDA environment, so you must activate it to launch the service).
If after inserting your password you are in, we are done (with PostgreSQL)!

Now you can configure stune. Please run `python -m stune.config` in the current project directory to initialize it. You can pass the parameter `--fix` to validate and fix your JAX installation (only works if using a conda environment). Use your-username your-password and hostname (if by any reasons your psql server is shutdown and is restarted on a different node, you will have to reconfigure stune by updating the hostname to match the new one: check `python -m stune.config --help`).

NOTE: At the moment stune does not support neptune.ai, so no need to configure it (simply leave the configuration fields blank).
