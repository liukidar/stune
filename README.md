# STUNE: Slurm Tuning Utility
Stune is based on Optuna and simplifies its integration with a SLURM cluster.

### Installation and configuration: PSQL
In order to use stune, you will need a working postgres sql server accessible by compute nodes. It is possible to install it both as sudo (globally) or as local user.

#### SUDO USER

[TODO]

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
