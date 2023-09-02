import argparse
import os
import json

def get_env(env, key, show_old = False):
    value = input(f"{key}" + (f" ({env[key]})" if show_old and key in env else "") +": ")

    if value != "":
        env[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set credentials used by stune")
    parser.add_argument("-u", "--user", action="store_true", help="Ask database user credentials")
    parser.add_argument("-d", "--database", action="store_true", help="Ask database host")
    parser.add_argument("-n", "--neptune", action="store_true", help="Ask neptune.ai credentials")

    args = parser.parse_args()

    # Check if necessary folders exist
    if not os.path.exists(".stune"):
        os.mkdir(".stune")
        os.mkdir(".stune/output")
        os.mkdir(".stune/config")

    config_filename = ".stune/config.json"
    new_config = False
    try:
        with open(config_filename, "r") as f:
            env = json.load(f)
    except FileNotFoundError:
        env = {}
        new_config = True

    if args.user or new_config:
        get_env(env, "PSQL_USR", True)
        get_env(env, "PSQL_PWD")
    if args.database or new_config:
        get_env(env, "PSQL_HOST", True)
    if args.neptune or new_config:
        get_env(env, "NEPTUNE_PROJECT", True)
        get_env(env, "NEPTUNE_API_TOKEN", False)

    with open(config_filename, "w") as f:
        json.dump(env, f, indent=4)
