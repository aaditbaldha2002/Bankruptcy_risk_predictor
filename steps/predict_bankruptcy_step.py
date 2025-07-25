import logging
from zenml.steps import step
from src.inferencing.predict_on_cluster_label import predict_on_cluster_label

@step(enable_cache=True)
def predict_bankruptcy_step(transformed_data_file_path:str,cluster_label:int)->int:
    try:
        logging.info("Starting the predict_bankruptcy_step step ...")
        final_prediction=predict_on_cluster_label(transformed_data_file_path,cluster_label)
        logging.info("predict_bankruptcy_step completed")
        return final_prediction
    except Exception as e:
        logging.error(f"Error occurred in predict_bankruptcy_step step:{e}")
        raise e
    pass