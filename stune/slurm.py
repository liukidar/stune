from typing import Optional, List
import os
import subprocess

class Sbatch:
    def __init__(
        self,
        cmd: str,
        tasks_per_node: int = 1,
        cpus_per_task: int = 1,
        gpu_reserved_memory : float = 0.1,
        time_minutes: int = 60,
        job_name: str = "lxp14",
        gpus: Optional[int] = None,
        partition: str = "devel",
        output: Optional[str] = ".stune/output/%j-%x.out",
        env: Optional[str] = "base",
        ld_library_path: str = "",
        resources: Optional[List[str]] = None
    ):
        sbatch_filename = f"__sbatch_{job_name}.sh"
        self.sbatch_filename = sbatch_filename

        sbatch_cmd = "#!/bin/bash -l\n"
        sbatch_cmd += f"#SBATCH --nodes=1\n"
        sbatch_cmd += f"#SBATCH --tasks-per-node={tasks_per_node}\n"
        sbatch_cmd += f"#SBATCH --cpus-per-task={cpus_per_task}\n"
        sbatch_cmd += f"#SBATCH --time={time_minutes // 60}:{time_minutes % 60}:00\n"
        sbatch_cmd += f"#SBATCH --job-name={job_name}\n"
        if gpus is not None:
            sbatch_cmd += f"#SBATCH --gres=gpu:{gpus}\n"
        sbatch_cmd += f"#SBATCH --partition={partition}\n"
        if output is not None:
            sbatch_cmd += f"#SBATCH --output {output}\n"

        sbatch_cmd += "module purge\n"

        # Activate conda
        if env is not None:
            sbatch_cmd += "module load python/anaconda3 &>/dev/null\n"
            sbatch_cmd += "eval \"$(conda shell.bash hook)\"\n"
            sbatch_cmd += f"conda activate {env}\n"

        # LD_LIBRARY_PATH should include the cuda compability fix
        sbatch_cmd += f"export LD_LIBRARY_PATH=\"{ld_library_path}\"\n"

        # Set gpu memory fraction per task
        sbatch_cmd += f"export XLA_PYTHON_CLIENT_PREALLOCATE=true\n"
        sbatch_cmd += f"export XLA_PYTHON_CLIENT_MEM_FRACTION=\".{int(100 * (1.0 - gpu_reserved_memory*tasks_per_node) / tasks_per_node)}\"\n"

        # Copy over required dataset (TODO: should be changed to a set of requests made via the config file)
        if resources is not None:
            for resource in resources:
                sbatch_cmd += f"rsync -a $HOME/{resource} $TMPDIR\n"

        if tasks_per_node > 1:
            sbatch_cmd += "srun --ntasks $SLURM_NTASKS "
            if gpus is not None:
                sbatch_cmd += f"--gres=gpu:{gpus} "

        sbatch_cmd += cmd + "\n"

        with open(".stune/" + sbatch_filename, "w") as f:
            f.write(sbatch_cmd)
        os.system(f"chmod +x .stune/{self.sbatch_filename}")

    
    def __del__(self):
        os.system(f"rm .stune/{self.sbatch_filename}")
    
    def submit(self):
        return subprocess.Popen(["sbatch", "--wait", f".stune/{self.sbatch_filename}"])