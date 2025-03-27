# rosettaparsl
Code for scaling up Rosetta on the ALCF's Aurora supercomputer using Parsl.

## Installation

```shell
#Create the virtual environment.
mamba create -n rosettaparsl-env python=3.12 -y
mamba activate rosettaparsl-env

#Download and install pyRosetta
mkdir pyRosetta
cd pyRosetta

wget https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python312.ubuntu/PyRosetta4.Release.python312.ubuntu.release-395.tar.bz2

tar -vjxf PyRosetta4.Release.python312.ubuntu.release-395.tar.bz2

cd setup
python setup.py install

```