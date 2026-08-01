import os
import subprocess

root_dir = '/nobackup/dbrethau/GradientGridNeutron1D/'

dyn_mass = ['m3e3','m1e2','m3e2']

nmass = ['m1e7','m1e6','m1e5','m1e4']

ns = ['5','10','20','500']

all_dirs = [root_dir+f"models_n{n}/{net}/{dyn}/" for dyn in dyn_mass for net in nmass for n in ns]

for model_dir in all_dirs:
    job_name = model_dir[model_dir.find('models_n')+len('models_'):]
    job_name = job_name.replace("/","_")[:-1]

    qsub_cmd = [
        "qsub",
        "-N", job_name,
        "-v", f"TARGET_DIR={model_dir}",
        "job.pbs"
    ]
#    print(qsub_cmd)
    result = subprocess.run(qsub_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully submitted '{job_name}' -> Job ID: {result.stdout.strip()}")
    else:
	print(f"Failed to submit '{job_name}': {result.stderr}")
