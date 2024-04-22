# Demo scheduler
Implementation of uniform crossover and moc crossover,
from [A Two-stage Genetic Algorithm for a Novel FJSP with Working Centers
in a Real-world Industrial Application](https://www.scitepress.org/Papers/2021/106549/106549.pdf)

## Package structure
- demo_scheduler/crossover.py: implementation of MOC and uniform crossovers
- demo_scheduler/main.py: main file
- demo_scheduler/tests/corner_cases.py: corner cases
- demo_scheduler/tests/test_demo_scheduler.py: unit tests

## Install process
### Requirements
- pyenv: [pyenv-installer](https://github.com/pyenv/pyenv-installer#install)

### Installation steps
1. Clone the repository
2. Check and optionally install required Python version
```Bash
pyenv install $(cat .python-version)
```
3. Activate pyenv and create a virtualenv
```Bash
pyenv local
python -m pip install virtualenv
python -m virtualenv .venv
```
4. Activate virtualenv .venv
```Bash
. .venv/bin/activate
```
5. Poetry install
```Bash
./scripts/install-poetry.sh
```
6. Install required libraries  
```Bash
python -m poetry install
```

## How to run main
From root folder:
```Bash
python demo_scheduler/main.py
```

## How to run unit tests
From root folder:
```Bash
./scripts/test.sh
```
