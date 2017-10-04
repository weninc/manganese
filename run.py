import os
import sys
import time
import subprocess
import numpy as np

folder ='/Users/weninc/xrl/data/'
#folder ='/home/clemens/xrl-data'
max_processes = 2

nph = np.linspace(1.0e10, 1.0e12, 20)

def run():
    processes = {}
    for j in range(50):
        print(j)
        output_file = os.path.join(folder, 'data-%02d.h5' %j)
        p = subprocess.Popen(['./a.out', output_file])
        processes[p.pid] = p
        if len(processes) >= max_processes:
            pid, ret = os.wait()
            del processes[pid]

    # wait for them to finish
    while processes:
        pid, ret = os.wait()
        del processes[pid]

run()
