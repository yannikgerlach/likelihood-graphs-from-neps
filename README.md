# Inferring A Multi-Perspective Likelihood Graph from Black-Box Next Event Predictors
This repository contains a prototype implementation of the method described in the paper *Inferring A Multi-Perspective Likelihood Graph from Black-Box Next Event Predictors*.
The method infers a directed and acyclic likelihood graph from a (deep learning) next event predictor to reveal and visualize what it has learned.

## Abstract
Deep learning models for next event prediction in predictive process monitoring have shown significant performance improvements over conventional methods.
However, they are often criticized for being black-box models.
Without allowing analysts to understand what such models have learned, it is difficult to establish trust in their abilities.

In this work, we propose a technique to infer a likelihood graph from a next event predictor to capture and visualize its behavior.
Our approach first generates complete cases, including event attributes, using the next event predictor. 
From this set of cases, a multi-perspective likelihood graph is inferred.
Including event attributes in the graph allows to better understand the learned decision and branching points within a process.

The evaluation shows the inferred graphs generalize beyond the event log and achieve high F-scores and small likelihood deviations.
We conclude black-box NEPs can be used to generate correct cases even in the presence of anomalies in the event log.
As a result, our visualization technique, which represents exactly this set of cases, shows what the NEP has learned, thus mitigating one of their biggest criticisms.

## Example Likelihood Graphs
The following example likelihood graph was generated using our approach. More can be found in `/examples`.

![](examples/small.jpg)

## Jupyter Lab Notebooks
Check the `notebooks` directory for example Jupyter Lab Notebooks. 
Follow the setup guide and make sure to install the *ipywidgets* extension (https://ipywidgets.readthedocs.io/en/latest/user_install.html) for Jupyter Lab.

## Setup
The easiest way to setup an environment is to use Miniconda.

### Using Miniconda
1. Install [Miniconda](https://conda.io/miniconda.html).
2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash).
3. We suggest that you set up a dedicated environment for this project by running `conda env create -f environment.yml`
    * This will setup a virtual conda environment with all necessary dependencies.
    * Installs the CPU only version of Tensorflow. 
4. Depending on your operating system you can activate the virtual environment with `conda activate neplg`.
6. If you want to quickly install the `neplg` package, run `pip install -e .` inside the root directory.
7. Now you can start the notebook server by `jupyter lab notebooks`.

Note: To use the graph plotting methods, you will have to install [Graphviz](https://graphviz.org/).

### IDE
We recommend you use [PyCharm](https://www.jetbrains.com/pycharm) Community Edition as your IDE.

## References
* [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)
* [Nolle, T., Luettgen, S., Seeliger, A., Mühlhäuser, M.: BINet: Multi-perspective Business Process Anomaly Classification, 2019](https://doi.org/10.1016/j.is.2019.101458)