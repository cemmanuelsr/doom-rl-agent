# Doom Reinforcement Learning Agent

A reinforcement learning agent developed on the [VizDoom environment](http://vizdoom.cs.put.edu.pl/).

## Setup

1. Install dependencies following the [official documentation](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)
2. `pip install -r requirements.txt`
3. Run `python basic.py` to test if installation was successful

## How to control the script

The main script is [agent.py](https://github.com/cemmanuelsr/doom-rl-agent/blob/master/agent.py), you can control what it runs using the [constants.py](https://github.com/cemmanuelsr/doom-rl-agent/blob/master/constants.py). All the parameters are well described by their names, but {number} are most important:

- `model_savefile`: name of the file which model will be (was) saved;
- `save_model`: instruction to save model or just train and watch a new agent;
- `load_model`: instruction to load an existent model or not;
- `visible_during_train`: control if game image will appear during train or not (keep it False to speed up train process);
- `show_labels`: in future will control if agent use labels extracted with OpenCV.

If you want to train a new agent, set `save_model = True` and `load_model = False`. To run a existent model, set `save_model = False` and `load_model = True` (and choose the apropriate model name).

## Observations for training

- According to the official documentation from ViZDoom, 5-10 epochs are okay to train a good model. I discovered that it is only true for some scenarios (e.g. basics modified scenarios), but I couldn't find a good number for all scenarios;
- Agents trained with a large number of epochs (50 or higher) start to decrease its learning capabilities for misterious reasons;
- Agents are saved in agents folder.
- After trained, if `save_model = True`, the script will generate a "Score x Epochs" graph on results folder.
