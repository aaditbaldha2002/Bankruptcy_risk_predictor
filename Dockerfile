FROM public.ecr.aws/lambda/python:3.10

# Set working directory
WORKDIR /app

COPY src/apis/predict/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["src.deployments.lambda.lambda_handler.handler"]