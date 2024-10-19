---
title: UFC Predictor
emoji: 🥋
colorFrom: blue
colorTo: red
sdk: docker
python_version: 3.12
pinned: false
---

# UFC Predictor

[![Python application](https://github.com/balaustrada/ufcpredictor/actions/workflows/python-app.yml/badge.svg)](https://github.com/balaustrada/ufcpredictor/actions/workflows/python-app.yml/)
[![Coverage Status](https://coveralls.io/repos/github/balaustrada/ufcpredictor/badge.svg?branch=main)](https://coveralls.io/github/balaustrada/ufcpredictor?branch=main)
[![mypy](https://github.com/balaustrada/ufcpredictor/actions/workflows/mypy.yml/badge.svg)](https://github.com/balaustrada/ufcpredictor/actions/workflows/mypy.yml)
[![mkdocs](https://github.com/balaustrada/ufcpredictor/actions/workflows/mkdocs.yml/badge.svg)](https://github.com/balaustrada/ufcpredictor/actions/workflows/mkdocs.yml)

Documentation available [here](https://balaustrada.github.io/ufcpredictor/)

Model deployed in this Hugging Face [space](https://huggingface.co/spaces/balaustrada/UFCPredictor)

# The Model
The model uses the statistics from previous matches as features to predict the outcomes of upcoming fights. It consists of two stacked neural networks. Key statistics for each fighter are input into the first network, producing two vector outputs. 

These outputs are then fed into the SymmetricFightNet, yielding a value between 0 and 1, where 0 indicates that the first fighter is predicted to win and 1 indicates the opposite. The loss function applied to the model optimises profit based on the odds input into the network, making the odds a crucial part of the model.

The output value of the network should be considered as a "normalised" predicted bet, which is scaled to the range (-100, 100) in the Gradio implementation.


# Predictions
In ``examples`` the notebooks ``results_*`` show the fight forecast for the periods 2023/01-2024/10 and 2024/01-2024/10. Two different models are shown:
- **With odds**: This is the default model, where the odds are used in the loss function, but they are also used as input features in the model.
- **Without odds**: This model is trained using the same loss function, but the odds do not enter into the model.

The accuracy of both models on these datasets is summarized in the table below, where Favorite Accuracy is the accuracy if we just predict the favorite to win.

<div align="center">
  
|   Period  | Odds in model | Accuracy (%) | Favourite Accuracy (%) |
|:---------:|:-------------:|:------------:|:---------------------:|
|    2024   |       No      |     60.3     |          64.0         |
|    2024   |      Yes      |     64.0     |          64.0         |
| 2023-2024 |       No      |     58.6     |          62.1         |
| 2023-2024 |      Yes      |     61.8     |          62.1         |

</div>

The accuracy of the model is comparable to that implied by the opening odds, although slightly lower. When odds are incorporated into the model, the accuracy aligns with that of predicting favourites. However, the difference in predictions can still lead to profit, depending on the strength of the prediction and the odds. In the following plot, we compare the return for the different datasets:

<p align="center">
<img src="https://github.com/user-attachments/assets/d33a1412-dbb4-4358-88ad-f1789edf7e53">
</p>

while the model shows underperformance in certain periods, leading to loses, its overall performance shows potential for profit. When evaluating return, we see an improvement with the inclusion of odds in our model; but the behaviour of both curves is similar.

# Deployment
The app is deployed in the following Hugging Face [space](https://huggingface.co/spaces/balaustrada/UFCPredictor), allowing for the prediction of fights by including the event date, fighter names and the odds assigned for each fighter.

If you want to deploy you own, you can follow the following guidelines:

## Input data

In order to successfully deploy the prediction model, data needs to be downloaded and stored in a local folder. This data is expected to follow the structure defined in [balaustrada/ufcscraper](https://github.com/), with the following tables:
- ``event_data.csv``
- ``fighter_data.csv``
- ``fight_data.csv``
- ``round_data.csv``
- ``fighter_names.csv``
- ``BestFightOdds_odds.csv``

Alternatively, data can be downloaded on the fly if a Hugging Face token to the repo ``datasets/UFCfightdata`` is provided (currently a private repo).

## Local installation
Install ``ufcscraper`` and ``ufcpredictor`` (tested for ``python==3.12``):
```bash
RUN pip install git+https://github.com/balaustrada/ufcscraper.git
RUN pip install git+https://github.com/balaustrada/ufcpredictor.git
```

### Run (without UFCfightdata token)
In this case you need to specify the data folder where the ``csv`` files are stored, and the model: 
```bash
ufcpredictor_app --data-folder data --model-path models/model.pth
```
The model can be accessed in ``http://localhost:7860``.


### Run (with UFCfightdata token)
In this case you need specify the data folder where the data will be downloaded into, but also the token and the model:
```bash
export DATASET_TOKEN=DATASETTOKEN; ufcpredictor_app --data-folder data --model-path models/model.pth --download-dataset
```
The model can be accessed in ``http://localhost:7860``.


## Deployment with Docker

### Include data (without UFCfightdata token)
We need to modify the Dockerfile uncommenting and editing the following line to include the folder where the ``csv`` files are stored. (Remember Dockerfile will only work with folders inside the relative path).
```dockerfile
#ADD --chown=user data_folder $HOME/app/data
```
We should also modify the last line to remove the ``--download-dataset`` option:
```dockerfile
CMD ["python", "app.py", "--data-folder", "data", "--model-path", "models/model.pth", "--port", "7860", "--server-name", "0.0.0.0",]
```

### Include data (with token)
To include the data we only to set the environmental variable ``DATASET_TOKEN`` in our Dockerfile or at runtime.

### Build image
The first step to deploy with Docker is to **build the image**. We need first to modify the Dockerfile uncommenting and editing the line:
```bash
docker build -t ufcpredictor:latest .
```

### Run container (without token)
```bash
docker run -p 7860:7860 ufcpredictor
```
The model can be accessed in ``http://localhost:7860``.

### Run container (with token)
```bash
docker run -e DATASET_TOKEN=DATASET_TOKEN -p 7860:7860 ufcpredictor
```
The model can be accessed in ``http://localhost:7860``.
