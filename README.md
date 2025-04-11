# rosettaparsl
Code for scaling up Rosetta on the ALCF's Aurora supercomputer using Parsl.

## Installation
Instructions below assumes the following:

* Linux Ubuntu 64-bit OS
* Python 3.12 (in a virtual env)

If your config is different, download a different version of pyRosetta.

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

#Install the rest of the dependencies for rosettaparsl
pip install -e .

```

## Usage
From the root directory of the repository.

```shell
nohup python -m rosettaparsl.main --config examples/<your_config.yaml> &
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks. **Perform these after installing pyRosetta.**

```bash
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```