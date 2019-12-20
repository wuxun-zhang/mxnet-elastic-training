
import os
import glob
import mxnet as mx
import subprocess
import argparse
from threading import Thread

parser = argparse.ArgumentParser(description='Train a model for image classification.')

parser.add_argument('--schedule', type=str, default='./node_list')
parser.add_argument('--command', type=str, default='')
parser.add_argument('--model', type=str, default='resnet50_v1')
parser.add_argument('--data-dir', type=str, default=os.path.expanduser('~/distributed_training/datasets/imagenet/imagenet_pass_through'))
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--warmup-epochs', type=int, default=5)
parser.add_argument('--data-nthreads', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr-mode', type=str, default='cosine')
parser.add_argument('--mode', type=str, default='module',
                     help='module or gluon')
parser.add_argument('--epochs-per-update', type=int, default=3,
                    help='number of epoche during one elastic update period')
parser.add_argument('--fix-global-batch-size', type=int, default=-1,
                    help='fix the global batch size')

def parse_config(filename):
    schedule = {}
    if not filename:
        return schedule
    filename = os.path.expanduser(filename)
    with open(filename, 'r') as f:
        for line in f.readlines():
            epoch, num_gpus = line.split()
            schedule[int(epoch)] = int(num_gpus)
    return schedule

class DeviceSchedule:
    def __init__(self, config):
        self.schedule = parse_config(config)

    def detect_change(self, epoch):
        return epoch in self.schedule

    def num_node(self, epoch):
        return self.schedule[epoch]

def run(prog):
    subprocess.check_call(prog, shell = True)

def check_end(epoch):
    params = './model/%s-0-%04d.params' % (args.model, epoch)
    return os.path.exists(params)

args = parser.parse_args()
device_schedule = DeviceSchedule(args.schedule)

def elastic():
    num_process = 4 # two nodes
    for epoch in range(args.epochs):
        if device_schedule.detect_change(epoch):
            num_process = device_schedule.num_node(epoch)
        if check_end(epoch + 1):
            continue
        # get env from system
        path = os.getenv('PATH')
        pythonpath = os.getenv('PYTHONPATH')
        ld_library_path = os.getenv('LD_LIBRARY_PATH')
        prog = 'export PATH=' + ('' if path is None else path) + ';export PYTHONPATH=' + ('' if pythonpath is None else pythonpath) + \
               ';export LD_LIBRARY_PATH=' + ('' if ld_library_path is None else ld_library_path) + ';export MXNET_ENGINE_TYPE=NaiveEngine;' + \
               'export MLSL_USE_MPI_FORCE=1;'
        # launch mpi program
        prog = prog + 'mpirun -n ' + str(num_process) + ' -ppn 2 -f hostfile -genv I_MPI_ASYNC_PROGRESS=1 -genv I_MPI_PIN_DOMAIN=[0x00000ffffe0000000000,0xffffe000000000000000]' + \
               ' -genv I_MPI_ASYNC_PROGRESS_PIN=0 -genv I_MPI_ASYNC_PROGRESS_THREADS=1 -genv MXNET_USE_OPERATOR_TUNING=0 -genv OMP_NUM_THREADS=18' + \
               ' -genv HOROVOD_FUSION_THRESHOLD=0 -genv I_MPI_DEBUG=4 '
        prog = prog + args.command
        prog = prog + ' --model=' + args.model
        prog = prog + ' --use-rec'
        prog = prog + ' --data-nthreads=' + str(args.data_nthreads)
        prog = prog + ' --rec-train=' + '/'.join([args.data_dir, 'train.rec'])
        prog = prog + ' --rec-train-idx=' + '/'.join([args.data_dir, 'train.idx'])
        prog = prog + ' --rec-val=' + '/'.join([args.data_dir, 'val.rec'])
        prog = prog + ' --rec-val-idx=' + '/'.join([args.data_dir, 'val.idx'])
        prog = prog + ' --batch-size=' + str(args.batch_size)
        prog = prog + ' --lr=' + str(args.lr)
        prog = prog + ' --lr-mode=' + args.lr_mode
        prog = prog + ' --warmup-epochs=' + str(args.warmup_epochs)
        prog = prog + ' --mode=' + args.mode

        prog = prog + ' --begin-epoch=' + str(epoch)
        prog = prog + ' --fix-global-batch-size=' + str(args.fix_global_batch_size)
        prog = prog + ' --epochs-per-update=' + str(args.epochs_per_update)
        prog = prog + ' --eval-epoch'
        prog = prog + ' --no-cuda'
        # prog = prog + ' --use-pretrained'
        print(prog)
        thread = Thread(target = run, args=(prog,))
        thread.setDaemon(True)
        thread.start()
        thread.join()
        #break
        while not(check_end(epoch + 1)):
            pass

if __name__ == '__main__':
    elastic()
