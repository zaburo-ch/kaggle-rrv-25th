import subprocess
import re
import numpy as np


np.random.seed(1024)


def extract_best_epoch(s):
    try:
        best_epoch = int(re.search(r'best_epoch:\s(\d+)', s).group(1))
    except:
        best_epoch = 300
    return best_epoch


def try_running(cmd):
    print(cmd, flush=True)
    try:
        res = subprocess.check_output(cmd, shell=True)
        res = res.decode('utf-8')
    except KeyboardInterrupt:
        print('Interrupted!', flush=True)
        raise KeyboardInterrupt
    except:
        res = 'fail!!!!'
    print(res, flush=True)
    return res


base_cmd = 'python run.py --model_name wavenet --cut_peak'

for i in range(5):
    seed = np.random.randint(0, 10000)
    for x_len in [28, 56]:
        cmd = base_cmd + ' --seed {} --x_len {}'.format(seed, x_len)
        res = try_running(cmd)

        nb_epoch = extract_best_epoch(res)
        cmd += ' --nb_epoch {} --without_valid'.format(nb_epoch)
        res = try_running(cmd)
