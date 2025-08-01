from mangum import Mangum
from src.apis.predict.main import app

handler=Mangum(app)