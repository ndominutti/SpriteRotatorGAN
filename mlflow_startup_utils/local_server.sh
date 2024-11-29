#!/bin/bash

read -sp "Enter AWS S3 uri: " bucket

mlflow server \
    --host 0.0.0.0 \
    --port 8080 \
    --default-artifact-root $bucket

echo "Starting MLflow server at http://localhost:8080..."