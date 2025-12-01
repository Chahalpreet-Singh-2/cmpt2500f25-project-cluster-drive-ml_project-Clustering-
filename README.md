# ML_project
GoAuto best selling regions -Team Members: Chahalpreet Singh, Parminder Singh, Rajveer Singh and Arpandeep kaur. PROJECT TITLE: GoAuto Best selling regions Welcome to the repository for our project at Norquest College. This project aims to look for best selling regions where go auto sells more vehicles.
# The repository contains the following files:
configs- YMAL for training/predicting, data- raw and processed datasets, models- #trained model files, notebooks- jupyter notebooks for eda and prototyping src
# Running model
install dependencies- pip install- r requirements.txt preprocess data- python src/preprocess.py training model- python src/train.py evaluate performance- python src/evaluate.py
# Configurations
configs/train_config.yaml — for training hyperparameters and configs/predict_config.yaml — for prediction setup can be edited as per need.
# Using notebooks
Notebooks folder can be used for EDA, visulizations and model comparisons.
## Running the Containerized Application

### Prerequisites
- Docker
- Docker Compose

### Steps
Clone the repo and cd into it:
   
   git clone https://github.com/Chahalpreet-Singh-2/cmpt2500f25-project-cluster-drive-ml_project-Clustering-
   cd cmpt2500f25-project-cluster-driver-Lab2

### 1. Build and run services
docker-compose up --build

### 2. Run training inside the container
docker-compose run --rm ml-app python -m src.train

### 3. Docker Hub Images
- ML Application: `https://hub.docker.com/r/chahalpreetsingh/ml-application`
- MLflow Tracking: `https://hub.docker.com/r/chahalpreetsingh/mlflow-tracking`

### Team Contributions

- Chahalpreet Singh: MLflow integration
- Parminder Singh: Dockerfiles (ml-app & mlflow)
- Rajveer Singh: docker-compose setup
- Arpandeep Kaur: docker login and push

