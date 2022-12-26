# You must set the value of the `SLURM_PARTITION` environment variable to run this
# script:
#
#
# Use the `sinfo` command to find the list of avialable SLURM partitions.
# More details of the `run_slurm.sh` script and environment variables it uses are here:
# https://ghe.oculus-rep.com/FRL-Graphics-Research/ml-tools
# --outdir is just an example - you are free to determine your output directory and how
# you pass it into your non-mpi application as you see fit.

# good nodes:
# sea112-dgx2-a10-07
# sea112-dgx2-a10-27
# sea112-dgx2-a11-27
# sea112-dgx2-a12-27

export EXTRA_SLURM_CMDS="--exclude=sea112-dgx[1001-1115],sea112-dgx2-a09-07,sea112-dgx2-a11-27"
export SLURM_PARTITION="GraphicsTeam"
export DOCKER_IMAGE_NAME="pytorch17:2021-01017"
export NUM_NODES=1
export GPUS_PER_TASK=1
export CPUS_PER_TASK=1
export RANKS_PER_TASK=1
export USE_MPI=0


for res in 1280x720 1920x1080 3840x2160
do
    for f in 1 60
    do
        echo JOB_NAME="vdp_bench_${res}x${f}" $(pwd -P)/../ml-tools/run_slurm.sh python3 $(pwd -P)/metric.py --gpu 0 --benchmark ${res}x${f} $@
    done
done

