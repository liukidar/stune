from typing import Optional, List
import os
import subprocess
import random


class Sbatch:
    def __init__(
        self,
        cmd: str,
        tasks_per_node: int = 1,
        cpus_per_task: int = 1,
        time_minutes: int = 60,
        job_name: str = "lxp14",
        gpus: Optional[int] = None,
        partition: str = "devel",
        output: Optional[str] = ".stune/output/%j-%x.out",
        env: Optional[str] = "base",
        ld_library_path: str = "",
        resources: Optional[List[str]] = None
    ):
        sbatch_cmd = "#!/bin/bash -l\n"
        sbatch_cmd += "#SBATCH --nodes=1\n"
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

        # Copy over required dataset
        if resources is not None:
            for resource in resources:
                sbatch_cmd += f"rsync -a $HOME/{resource} $TMPDIR\n"

        if tasks_per_node > 1:
            sbatch_cmd += "srun --ntasks $SLURM_NTASKS "
            if gpus is not None:
                sbatch_cmd += f"--gres=gpu:{gpus} "

        sbatch_cmd += cmd + "\n"

        self.job_name = job_name
        self.sbatch_cmd = sbatch_cmd

    def submit(self, n_jobs: int = 1):
        sbatch_filename = f"__sbatch_{self.job_name}_{random.randint(1, 9999)}.sh"
        with open(".stune/" + sbatch_filename, "w") as f:
            f.write(self.sbatch_cmd)
        os.system(f"chmod +x .stune/{sbatch_filename}")

        # Temporary fix for SLURM bug (see: https://bugs.schedmd.com/show_bug.cgi?id=14298)
        os.environ.pop("SLURM_CPU_BIND", None)

        r = subprocess.run(["sbatch", f"--array=1-{n_jobs}", f".stune/{sbatch_filename}"])

        os.system(f"rm .stune/{sbatch_filename}")

        return r
