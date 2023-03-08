# BattSim

## Overview

BattSim.m is a MATLAB algorithm developed by [Dr. Bala](https://www.uwindsor.ca/engineering/electrical/343/dr-bala-balasingam) at the University of Windsor. Although [his github](https://github.com/SingamLabs) does not have the algorithm, it will be linked anyways.

I have taken this algorithm, and using demoBatterySimulator.m, modified it so it can be interacted with through `stdin` and `stdout`.
Using stdio as a pipe, and using the `subprocess` python module, BattSim.py hosts a `BattSim` class that can be used to simulate a battery, using Dr. Bala's algorithm.

**All credit for the algorithm goes to Dr. Bala.** I have simply created an interface to use his algorithm in Python.

## Development Status

The current version of this algorithm uses the MATLAB backend running in GNU Octave to simulate the battery itself. As of the most recent version, the current simulation has been ported over to the Python backend, but has not been completely fixed yet, as there were some bugs in MATLAB that were ported over to the python version.

The current simulation backend is being updated on the CurrentSim_Py branch.
