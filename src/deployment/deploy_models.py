import logging
import subprocess



def deploy_models(deploy:bool,model_uri:str)->None:

    if not deploy:
        logging.warning("🚫 Deployment flag is False. Skipping deployment.")
        return None

    try:
        logging.info("🚀 Deployment condition met. Starting MLflow model server...")

        process = subprocess.Popen(
            ['mlflow', 'models', 'serve', '-m', model_uri, '--no-conda', '-p', '5001'],
            stdout=subprocess.DEVNULL,  # or a log file
            stderr=subprocess.DEVNULL
        )
        logging.info("✅ MLflow server launched in the background.")
        return "http://127.0.0.1:5001"
    except Exception as e:
        logging.exception("❌ Deployment failed.")
        return None
