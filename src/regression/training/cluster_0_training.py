

from src.regression.train_model_for_cluster import register_trainer


@register_trainer(0)
def cluster_0_training(data_path:str)->str:
    pass