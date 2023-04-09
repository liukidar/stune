from typing import Optional
import os

def sbatch(
    cmd: str,
    n_jobs: int = 1,
    tasks_per_node: int = 1,
    cpus_per_task: int = 1,
    time_hours: int = 0,
    time_minutes: int = 1,
    job_name: str = "lxp14",
    gpus: Optional[int] = None,
    partition: str = "devel",
    output: Optional[str] = ".stune/output/%j-%x.out",
    env: Optional[str] = "base",
    cuda: bool = False,
    wait: bool = False
):
    sbatch_cmd = "#!/bin/bash -l\n"
    sbatch_cmd += f"#SBATCH --nodes=1\n"
    sbatch_cmd += f"#SBATCH --tasks-per-node={tasks_per_node}\n"
    sbatch_cmd += f"#SBATCH --cpus-per-task={cpus_per_task}\n"
    sbatch_cmd += f"#SBATCH --time={time_hours}:{time_minutes}:00\n"
    sbatch_cmd += f"#SBATCH --job-name={job_name}\n"
    if gpus is not None:
        sbatch_cmd += f"#SBATCH --gres=gpu:{gpus}\n"
    sbatch_cmd += f"#SBATCH --partition={partition}\n"
    if output is not None:
        sbatch_cmd += f"#SBATCH --output {output}\n"

    sbatch_cmd += "module purge\n"

    # Activate conda
    if env is not None:
        sbatch_cmd += "module load python/anaconda3\n"
        sbatch_cmd += "eval \"$(conda shell.bash hook)\"\n"
        sbatch_cmd += f"conda activate {env}\n"
    sbatch_cmd += "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH\n"

    if tasks_per_node > 1:
        sbatch_cmd += "srun --ntasks $SLURM_NTASKS "
        if gpus is not None:
            sbatch_cmd += f"--gres=gpu:{gpus} "

    sbatch_cmd += cmd + "\n"

    sbatch_filename = f"__sbatch_{job_name}.sh"
    with open(".stune/" + sbatch_filename, "w") as f:
        f.write(sbatch_cmd)

    os.system(f"chmod +x .stune/{sbatch_filename}; sbatch --array=1-{n_jobs} {'--wait' if wait else ''} .stune/{sbatch_filename}")

    os.system(f"rm .stune/{sbatch_filename}")
    
    return
