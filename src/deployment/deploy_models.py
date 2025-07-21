import logging
import subprocess



def deploy_models(deploy:bool,model_uri:str)->None:

    if not deploy:
        logging.warning("🚫 Deployment flag is False. Skipping deployment.")
        return None

    try:
        logging.info("🚀 Deployment condition met. Starting MLflow model server...")

        process = subprocess.Popen(
            ['mlflow', 'models', 'serve', '-m', model_uri, '-p', '5001'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        stdout, stderr = process.communicate(timeout=15)
        logging.info(f"[MLflow Serve STDOUT]:\n{stdout}")
        logging.error(f"[MLflow Serve STDERR]:\n{stderr}")

        return "http://127.0.0.1:5001"
    except Exception as e:
        logging.exception("❌ Deployment failed.")
        return None
