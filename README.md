Strong inductive biases provably prevent harmless interpolation
================================================================
This repository contains the official code of the ICLR 2023 paper

**[Strong inductive biases provably prevent harmless interpolation](https://openreview.net/forum?id=7i6OZa7oij)**

by Michael Aerni, Marco Milanta, Konstantin Donhauser, Fanny Yang.

Please cite our work and this code as

    @inproceedings{Aerni23,
      title={Strong inductive biases provably prevent harmless interpolation},
      author={Michael Aerni and Marco Milanta and Konstantin Donhauser and Fanny Yang},
      booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2023},
    }

Setup and dependencies
----------------------

### Initial setup
1. Install conda (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).
2. Create an environment via `conda env create -f environment.yml InductiveBiasesHarmlessInterpolation` (this might take some time).
3. Copy `template.env` to `.env` and update the entries where necessary.


### Using the environment
The environment can be enabled and disabled
from the command line as follows:

    # enable
    conda activate InductiveBiasesHarmlessInterpolation

    # disable
    conda deactivate


Experiments
-----------
For each experiment subset, we provide a bash script to run all the different settings,
and collect the results via [MLFlow](https://mlflow.org/).
We further provide Jupyter notebooks to plot experiment results and theoretical rates.


### Running
Each `run_*.sh` script in the `config/` directory
runs all configurations of a corresponding experiment subset.
Those scripts exactly reproduce our results without any further required actions.
We use [Gin Config](https://github.com/google/gin-config) to configure all settings.
Experiment parameters can be modified by either changing a `*.gin` file
or providing bindings explicitly via command line;
see the `run_*.sh` scripts for reference.
Finally, all our experiments run on a single consumer GPUs.
For the filter size experiments,
we use various GPUs with around 10GB of memory,
and NVIDIA GeForce RTX 2080 Ti for the rotational invariance experiments.


### Evaluation
The Jupyter notebooks in the `plots/` directory evaluate experimental runs.
Concretely, `filter_size.ipynb` and `rotations.ipynb` evaluate all
filter size and rotational invariance experiments, respectively,
including ablation experiments.
Lastly, `theory.ipynb` plots our theoretical rates in the paper.


### Environment variables
We simplify working with environment variables by using the `python-dotenv` package.
Hence, environment variables can be overwritten in an `.env` file,
placed in the root of this repository.
The file `template.env` serves as a template.


Datasets
--------
All filter size experiments use synthetic data that is generated ad-hoc (see `data.py`).
Note that dataset generation may take a long time, hence generated datasets can be cached.

The rotational invariance experiments use the [EuroSAT](https://github.com/phelber/EuroSAT)
dataset as provided by PyTorch.
By default, the experiments store raw EuroSAT data in a `datasets/` folder at the root
of this repoistory.
The location of this directory can be changed via the `DATA_DIR` environment variable.
