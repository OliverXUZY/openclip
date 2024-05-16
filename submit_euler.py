import subprocess
import shlex
import re
import os
import argparse

def run_sbatch_Euler(cmd, job_name, args):
    print(cmd + '\n')
    if args.print_command:
        return

    read_file = "./euler.sh"
    f = open(read_file)
    text = f.read()
    f.close()

    text = text.replace("run_command", cmd)
    text = text.replace("job_name", job_name)
    if args.research:
        text = text.replace("#SBATCH -p lianglab", "#SBATCH -p research")
    elif args.cpu:
        text = text.replace("#SBATCH --gres=gpu:1          ## GPUs", "")
    elif args.both:
        text = text.replace("#SBATCH -p lianglab", "#SBATCH -p lianglab,research")
    else:
        pass

    gpu = args.gpu
    gpu_command = "#SBATCH --gres=gpu:<GPUname>:<GPUnum>"
    gpu_command = gpu_command.replace("<GPUnum>", str(args.n_gpu))
    if gpu == "":
        gpu_command = gpu_command.replace(":<GPUname>", gpu)
    else:
        gpu_command = gpu_command.replace("<GPUname>", gpu)
    text = text.replace("#SBATCH --gres=gpu:1", gpu_command)

    if args.n_gpu == 4:
        print("adjust CPU and ram for 4 GPUs")
        # Replace CPUs per task
        n_cpus = 48
        cpu_command = "#SBATCH --cpus-per-task={n_cpus}".format(n_cpus=n_cpus)
        text = text.replace("#SBATCH --cpus-per-task=12", cpu_command)
    
        # Replace memory settings
        memory = 240
        mem_command = "#SBATCH --mem={memory}GB".format(memory=memory)
        text = text.replace("#SBATCH --mem=60GB", mem_command)




    path = f"./{job_name}.sh"

    f = open(path, "w")
    f.write(text)
    f.close()

    slurm_cmd = "sbatch " + path
    
    output = subprocess.check_output(shlex.split(slurm_cmd)).decode('utf8')
    print(output)
    job_names = list(re.findall(r'\d+', output))
    assert(len(job_names) == 1)

    os.remove(path)
    return job_names[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--run_bash", type=str, default="run_cuda0.sh", help="select bash file to run")
    parser.add_argument("--print_command", action="store_true", default=False)
    parser.add_argument("--research", action="store_true", default=False)
    parser.add_argument("--both", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="", help="GPU name")
    parser.add_argument("-ng", "--n_gpu", type=int, default=1)
    parser.add_argument('--n_cpus', type=int, help='Number of CPUs per task', default=16)
    parser.add_argument('--memory', type=int, help='Memory in GB', default=80)

    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.run_bash, "r") as file:
        cmd = file.read()
    
    run_sbatch_Euler(cmd = cmd, job_name = args.run_bash[3:], args = args)
    
if __name__ == "__main__":
    main()
