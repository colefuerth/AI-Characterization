import subprocess
import numpy as np
from math import log10

ssh = subprocess.Popen(["octave", "sp.m"],
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       universal_newlines=True,
                       bufsize=0)

# Send ssh commands to stdin
# assert(ssh.stdout.read()  == 'coefficients: ')

battery = [
    -9.082, 103.087, -18.185, 2.062, -0.102, -76.604, 141.199, -1.117
]

# if ssh.stdout.read() != 'coefficients':
#     raise Exception('coefficients request not received')
ssh.stdin.write(f'{" ".join(map(str,battery))}\n')
ssh.stdin.close()

# Fetch output
for line in ssh.stdout:
    print('py:', line.strip())

