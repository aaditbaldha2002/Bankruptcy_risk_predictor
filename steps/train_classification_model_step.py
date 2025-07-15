import logging
from sklearn.base import ClassifierMixin
from zenml.steps import step

@step(enable_cache=True)
def train_classification_model_step(data_path:str)->ClassifierMixin:
    pass