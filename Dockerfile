# ----------- Builder stage -----------
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy only requirements first for caching
COPY src/apis/predict/requirements.txt ./requirements.txt

# Build wheels for dependencies (no cache to keep slim)
RUN pip wheel --wheel-dir=/wheels --no-cache-dir -r requirements.txt

# ----------- Final stage -----------
FROM public.ecr.aws/lambda/python:3.10

WORKDIR /app

# Copy wheels from builder and install without cache
COPY --from=builder /wheels /wheels
COPY src/apis/predict/requirements.txt ./requirements.txt
RUN pip install --no-index --find-links=/wheels --no-cache-dir -r requirements.txt && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Copy only needed source files, avoid copying everything blindly
COPY src/apis/ ./src/apis/
COPY src/__init__.py ./src/__init__.py

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["src.deployments.lambda.lambda_handler.handler"]
