# Improving Intensive Care Chest X-Ray Classification by Transfer Learning and Automatic Label Generation

Code baseline for the paper "Improving Intensive Care Chest X-Ray Classification by Transfer Learning and Automatic Label Generation", published at [ESANN 2022].

## Description

Radiologists commonly conduct chest X-rays for the diagnosis of pathologies or the evaluation of extrathoracic material positions in intensive care unit (ICU) patients. Automated assessments of radiographs have the potential to assist physicians by detecting pathologies that pose an emergency, leading to faster initiation of treatment and optimization of clinical workflows. The amount and quality of training data is a key aspect for developing deep learning models with reliable performance. This work investigates the effects of transfer learning on public data, automatically generated data labels and manual data annotation on the classification of ICU chest X-rays of a German hospital. A more detailed description of the procedure is available at [ESANN 2022].

## Installation

In order to set up the necessary environment via conda:

1. review and uncomment what you need in `environment.yml` and create an environment `key2med` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate key2med
   ```

or via pip:

1. install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. install `key2med` package:
   ```
   pip install -e .
   ```

## Running a training configuration

Configurations to run the training described in the paper on the private dataset are provided in `configs/`.
To run a configuration, adjust local paths in the `training_gold_labels.yaml` and `training_silver_labels.yaml` file and run
```
python scripts/train.py -c configs/config.yaml
```
The configuration `training_gold_labels.yaml` contains the training hyperparameters for the training on gold labels. The variable load_checkpoint can be used to determine whether a pre-trained network is loaded.
The configuration `training_silver_labels` is used to set the hyperparameters for training on silver labels.


## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations to reproduce training runs in the paper.
├── environment.yml         <- The conda environment file for reproducibility.
├── scripts                 <- Contains training script.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `pip install -e .` to install for development
├── src
│   └── key2med             <- Actual Python package where the main functionality goes.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.0.2

[conda]: https://docs.conda.io/
[PyScaffold]: https://pyscaffold.org/
[ESANN 2022]: https://www.esann.org/
