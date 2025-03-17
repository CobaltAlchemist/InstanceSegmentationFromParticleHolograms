# InstanceSegmentationFromParticleHolograms

This project, `InstanceSegmentationFromParticleHolograms`, provides tools for instance segmentation of particle holograms using deep learning models. It includes functionalities for generating synthetic hologram samples, training models, and running inference.

You can see the article [here](https://iopscience.iop.org/article/10.1088/1361-6501/adb50f) and preprint [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4960633)

## Installation

To set up the environment and install dependencies, follow these steps:

1. Install [detectron2](https://github.com/conansherry/detectron2/blob/master/INSTALL.md)
2. Install [maskdino](https://github.com/IDEA-Research/MaskDINO/blob/main/INSTALL.md)
    - Note: Use the setup script in scripts/ (maskdino_setup.py -> setup.py) to install maskdino
3. Install this repository:
```sh
pip install https://github.com/CobaltAlchemist/InstanceSegmentationFromParticleHolograms
```

## Usage

We provide two modules, fakeholo and holodino. Fakeholo has all the tools for synthesizing holograms and serving a dataset for training. Holodino has all the tools for training and running inference on the holodino model.

### Fakeholo

The fakeholo module can be used as a typical python module. It uses Typer for command line interface. You can run the module by calling the module and the function you want to run. For more information, see the help message

```sh
python -m fakeholo --help
```

### Holodino

Holodino follows detectron2 more closely and uses argparser. But you can also view the help for this and it'll tell you what to do for the most part

```sh
python -m holodino --help
```

### VSCode

We have some tasks set up for VSCode that you can use to run the code. You can find them in the `.vscode/tasks.json` file. You can run them by pressing `Ctrl+Shift+P` and typing `Tasks: Run Task` and then selecting the task you want to run.

These help guide you through the process of running the code and can be used to run the code without having to remember the commands.

### Configs

The configs are structured simply, you can see the tree below:

- base-holodino.yaml
    - holodino-large.yaml
    - holodino-small.yaml

These will be used when you're specifying --config for holodino. They provide a baseline for configuring an actual model. The models provided by us have their own configs that you'll be loading instead.

### Model folder

If you would like to download some models to play around with, you can find them at [this google drive link](https://drive.google.com/file/d/1nyRCUzuAaFH0y0Avsk4dscZLNX9BMDbb/view?usp=sharing)

You should expand them such that you have this structure:

```
models/
    ├── holodino/
    │   ├── config.yaml
    │   ├── model_final.pth
    ├── dentalparticles/
    │   ├── config.yaml
    │   ├── model_final.pth
    ├── waterdroplets/
    │   ├── config.yaml
    │   ├── model_final.pth
```

### Examples

For Fakeholo, you can run the generative server standalone like this:

```sh
python -m fakeholo server
```

To generate some samples from the a dataset config, you can run this:

```sh
python -m fakeholo generate -n 10 --config datasets/fakeholo_hard/cfg.json -o ./samples
```

If you'd like to train a small holodino, you just have to run:

```sh
python -m holodino --config base-holodino.yaml --config holodino-small.yaml --train
```

`holodino` can also do evaluation and anything else you might want. Just check the .vscode/tasks for more examples