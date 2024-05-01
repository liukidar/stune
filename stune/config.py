import argparse
import os
import omegaconf


def get_env(env, key, show_old: bool = False):
    value = input(f"{key}" + (f" ({env[key]})" if show_old and key in env else "") + ": ")

    if (value != "") or (key not in env):
        env[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set credentials used by stune")
    args = parser.parse_args()

    # Check if necessary folders exist
    if not os.path.exists(".stune"):
        os.mkdir(".stune")
        os.mkdir(".stune/output")
        os.mkdir(".stune/config")

    config_filename = ".stune/config.yaml"
    try:
        config = omegaconf.OmegaConf.load(config_filename)
    except FileNotFoundError:
        # Default values
        config = omegaconf.OmegaConf.create({
            "STUNE_STORAGE": "redis",
            "GPU_MEM_RESERVED": "0.1"
        })

    get_env(config, "STUNE_USR", True)
    get_env(config, "STUNE_PWD")
    get_env(config, "STUNE_HOST", True)

    omegaconf.OmegaConf.save(config, config_filename)

    print("\nConfiguration complete.")
