import argparse
import os
import re
import json
import requests
from pathlib import Path


def get_env(env, key, show_old = False):
    value = input(f"{key}" + (f" ({env[key]})" if show_old and key in env else "") +": ")

    if value != "":
        env[key] = value


def cmd_execute(str):
    os.system(str + " > .out")

    with open(".out", "r") as f:
        output = f.read()
    os.remove(".out")

    return output

def sbatch_execute(cmd, ld_library_path):
    sbatch_cmd = "#!/bin/bash -l\n"
    sbatch_cmd += f"#SBATCH --nodes=1\n"
    sbatch_cmd += f"#SBATCH --time=1:00:00\n"
    sbatch_cmd += f"#SBATCH --gres=gpu:{1}\n"
    sbatch_cmd += f"#SBATCH --partition=devel\n"
    sbatch_cmd += f"#SBATCH --output ./.stuneconfig.out\n"

    sbatch_cmd += "module purge\n"

    # Activate conda
    sbatch_cmd += "module load python/anaconda3 &>/dev/null\n"
    sbatch_cmd += "eval \"$(conda shell.bash hook)\"\n"
    sbatch_cmd += f"conda activate {os.environ['CONDA_DEFAULT_ENV']}\n"

    if ld_library_path is not None:
        sbatch_cmd += f"export LD_LIBRARY_PATH=\"{ld_library_path}\"\n"

    sbatch_cmd += cmd + "\n"

    sbatch_filename = f".sbatch.sh"
    with open(sbatch_filename, "w") as f:
        f.write(sbatch_cmd)

    os.system(f"chmod +x {sbatch_filename}; sbatch --wait {sbatch_filename} >/dev/null")

    os.system(f"rm {sbatch_filename}")

    with open(".stuneconfig.out", "r") as f:
        output = f.read()
    os.remove(".stuneconfig.out")

    return output


def check_jax_installation(env):
    print("stune will now validate your JAX installation. This may take several minutes (or even more, based on how busy the server is), so don't panic if it looks stuck!")

    # Check if the installed version of JAX requires CUDA
    try:
        jaxlib_version = re.search("Version: (.*)", cmd_execute("pip show jaxlib")).group(1)
    except:
        jaxlib_version = None
    try:
        cuda_version = re.search("cuda(\d*)", jaxlib_version).group(1)
    except:
        cuda_version = None
    
    if jaxlib_version is None:
        print("JAX is not installed. Check here: https://github.com/google/jax")
        
        return None

    if cuda_version is None:
        print("The JAX version installed doesn't support CUDA. To fix this uninstall it and reinstall it through pip",
              " following the instructions found here under 'pip installation: GPU (CUDA, installed via pip, easier)'.\n",
              "NOTE: if you plan to use PyTorch this must be installed before JAX using, for example, the command:\n\n",
              "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n\n",
              "Make sure you use the same version of CUDA for both JAX and PyTorch (e.g., 11.8).")

        return None

    LD_LIBRARY_PATH = env.get("LD_LIBRARY_PATH", "")
    jax_output = sbatch_execute("python -c 'import jax; v = jax.numpy.zeros((1,1)); v@v'", LD_LIBRARY_PATH)

    # Check if gpu is detected, indicating that JAX is able to load CUDA
    if "CuDNN library needs to have matching major version and equal or higher minor version" in jax_output:
        print("\n\n[ERROR] JAX detected the wrong CUDA version, be sure to have installed a [cudaxx_pip] version of JAX.")
        
        return None
    
    # Check if cuda and driver versions are compatible
    elif ("use the NVIDIA-provided CUDA forward compatibility packages" in jax_output):
        c = input("\n\n[ERROR] stune detected that an old driver version is installed on the system. Do you want to attempt to fix the isse? \n\n"
                f"(This will install cuda compatibility drivers in the active conda environment, NO GLOBAL CHANGES will be made)\n\n? [y/n] ")

        if c in ["y", "Y"]:
            if args.cc_url:
                url = args.cc_url
                cuda_chosen = url.split("/")[-1]
            else:
                # Detect required cuda version
                required_cuda_version = re.search("ptxas CUDA version \((.*)\)", jax_output).group(1).split(".")

                # Download compatibility package
                print("\nAttempting to download conda compatibility package...")

                os_release = cmd_execute("cat /etc/os-release")
                os_name = re.search("NAME=\"(.*)\"", os_release).group(1)
                os_id = re.search("ID=\"(.*)\"", os_release).group(1)
                os_version = re.search("VERSION=\"(.*)\"", os_release).group(1)

                print(f"Detected operating system: {os_name} [{os_id} - {os_version}]")

                os_candidates = ([], [])

                for os_candidate in re.findall("href=[\"'](.+)/[\"']", requests.get("https://developer.download.nvidia.com/compute/cuda/repos/").text):
                    os_candidates[(os_candidate in os_id) or (os_id in os_candidate)].append(os_candidate)

                print("Available compatibility packages versions:\n")

                for os_candidate in os_candidates[0]:
                    print(os_candidate)
                
                print("\n[suggested]")
                for os_candidate in os_candidates[1]:
                    print(os_candidate)

                os_chosen = None
                while os_chosen is None:
                    c = input(f"\nSelect the compatibility package version that matches your OS [{os_id} - {os_version}]:\n")

                    if c in os_candidates[0] or c in os_candidates[1]:
                        os_chosen = c
                    else:
                        print("\nInvalid choice.")

                arch_candidates = re.findall("href=[\"'](.+)/[\"']",
                                            requests.get(f"https://developer.download.nvidia.com/compute/cuda/repos/{os_chosen}").text)

                arch_chosen = None
                if "x86_64" in arch_candidates:
                    arch_chosen = "x86_64"
                    print("\nx86_64 selected as default architecture.")
                else:
                    raise NotImplementedError()

                cuda_candidates = re.findall(f"href=[\"'](cuda-compat-{required_cuda_version[0]}-{required_cuda_version[1]}.+)[\"']",
                                            requests.get(f"https://developer.download.nvidia.com/compute/cuda/repos/{os_chosen}/{arch_chosen}").text)
                if len(cuda_candidates) == 0:
                    print("No valid cuda compatibility package found. You can manually find it by browsing"
                        f" 'https://developer.download.nvidia.com/compute/cuda/repos/' and specify its url through the --cc_url flag")
                    exit(0)
                elif len(cuda_candidates) == 1:
                    cuda_chosen = cuda_candidates[0]
                else:
                    pass
                
                url = f"https://developer.download.nvidia.com/compute/cuda/repos/{os_chosen}/{arch_chosen}/{cuda_chosen}"

            print("\n\nDownloading cuda compatibility package...")
            os.system(f"wget -O .stune-tmp/{cuda_chosen} {url}")

            print(f"\nExtracting {cuda_chosen} using rpm2cpio")
            os.system(f"cd .stune-tmp; rpm2cpio {cuda_chosen} | cpio -idmv")

            # Get directory containing the compatibility files
            cuda_compatibility_dir = list(Path(".stune-tmp").rglob("*.so*"))[0].parents[0]

            # Move compatibility files to conda_env/lib and adding them to LD_LIBRARY_PATH
            print(f"\nMoving cuda compatibility package to the active conda environment")
            os.system(f"mv {cuda_compatibility_dir} {os.environ['CONDA_PREFIX']}/lib/cuda-compat")
            LD_LIBRARY_PATH = f"{os.environ['CONDA_PREFIX']}/lib/cuda-compat/"

    return LD_LIBRARY_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set credentials used by stune")
    parser.add_argument("-u", "--user", action="store_true", help="Ask database user credentials")
    parser.add_argument("-d", "--database", action="store_true", help="Ask database host")
    parser.add_argument("-n", "--neptune", action="store_true", help="Ask neptune.ai credentials")

    parser.add_argument("--fix", action="store_true", help="Perform check of fix of the JAX installation.")
    parser.add_argument("--cc_url", type=str, help="Direct url from which to download the cuda compatibility package if necessary.")

    args = parser.parse_args()

    # Check if necessary folders exist
    if not os.path.exists(".stune"):
        os.mkdir(".stune")
        os.mkdir(".stune/output")
        os.mkdir(".stune/config")
    
    try:
        os.mkdir(".stune-tmp")
    except:
        pass

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
    
    if args.fix:
        env["LD_LIBRARY_PATH"] = check_jax_installation(env)
    else:
        # This is necessary until https://github.com/google/jax/issues/17497 is solved
        env["LD_LIBRARY_PATH"] = ""

    with open(config_filename, "w") as f:
        json.dump(env, f, indent=4)
    
    os.system("rm -rf .stune-tmp")
    print("\nConfiguration complete.")