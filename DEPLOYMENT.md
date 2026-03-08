# Deployment Roadmap: Transonic Aerodynamic Surrogate API

## Overview

After training and evaluating the surrogate models, the best-performing model is packaged into a REST API using FastAPI, containerized with Docker, and deployed to AWS ECS (Elastic Container Service) via ECR (Elastic Container Registry). The result is a scalable, production-style ML inference endpoint that accepts flight conditions and airfoil geometry as input and returns predicted Cl, Cd, and Cm in milliseconds.

---

## Stack

| Layer | Technology |
|---|---|
| Model serving | FastAPI |
| Containerization | Docker |
| Container registry | AWS ECR |
| Orchestration | AWS ECS (Fargate) |
| Load balancing | AWS ALB (Application Load Balancer) |
| Infrastructure | AWS CDK or Terraform (optional) |

---

## Phase 1: FastAPI Inference Service

Wrap the trained model in a FastAPI app with a single prediction endpoint.

**Endpoint:** `POST /predict`

**Request schema:**
```json
{
  "mach": 0.65,
  "aoa": 3.0,
  "geometry": [0.12, 0.02, 0.40, 0.008, 0.14, 0.32, 0.28]
}
```

**Response schema:**
```json
{
  "Cl": 0.734,
  "Cd": 0.0142,
  "Cm": -0.082,
  "model": "xgboost",
  "latency_ms": 0.4
}
```

**Implementation:**
```python
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import time

app = FastAPI(title="Transonic Aero Surrogate API")

# Load model and scaler at startup
model   = joblib.load("model/best_model.pkl")
X_scaler = joblib.load("model/X_scaler.pkl")
y_scaler = joblib.load("model/y_scaler.pkl")

class AeroInput(BaseModel):
    mach: float
    aoa: float
    geometry: list[float]

class AeroOutput(BaseModel):
    Cl: float
    Cd: float
    Cm: float
    latency_ms: float

@app.post("/predict", response_model=AeroOutput)
def predict(data: AeroInput):
    t0 = time.perf_counter()
    x = np.array([[data.mach, data.aoa] + data.geometry])
    x_scaled = X_scaler.transform(x)
    y_scaled = model.predict(x_scaled)
    y = y_scaler.inverse_transform(y_scaled)[0]
    latency = (time.perf_counter() - t0) * 1000
    return AeroOutput(Cl=y[0], Cd=y[1], Cm=y[2], latency_ms=latency)

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## Phase 2: Dockerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi
uvicorn
scikit-learn
xgboost
torch
numpy
pandas
joblib
pydantic
```

**Build and test locally:**
```bash
docker build -t aero-surrogate .
docker run -p 8000:8000 aero-surrogate

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"mach": 0.65, "aoa": 3.0, "geometry": [0.12, 0.02, 0.40, 0.008, 0.14, 0.32, 0.28]}'
```

---

## Phase 3: Push to AWS ECR

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS \
  --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository
aws ecr create-repository --repository-name aero-surrogate

# Tag and push
docker tag aero-surrogate:latest \
  <account_id>.dkr.ecr.us-east-1.amazonaws.com/aero-surrogate:latest

docker push \
  <account_id>.dkr.ecr.us-east-1.amazonaws.com/aero-surrogate:latest
```

---

## Phase 4: ECS Deployment (Fargate)

Fargate is serverless ECS — no EC2 instances to manage. AWS handles the underlying infrastructure.

**Steps:**
1. Create ECS cluster
2. Define task definition (CPU: 256, Memory: 512MB — sufficient for this model size)
3. Create ECS service with desired count = 1
4. Attach Application Load Balancer for public HTTPS endpoint
5. Configure auto-scaling on CPU utilization if needed

**Task definition (key fields):**
```json
{
  "family": "aero-surrogate",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [{
    "name": "aero-surrogate",
    "image": "<account_id>.dkr.ecr.us-east-1.amazonaws.com/aero-surrogate:latest",
    "portMappings": [{"containerPort": 8000}],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/aero-surrogate",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

---

## Phase 5: CI/CD with GitHub Actions

Automatically rebuild and redeploy on every push to `main`.

```yaml
# .github/workflows/deploy.yml
name: Deploy to ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build, tag, push image
        run: |
          docker build -t aero-surrogate .
          docker tag aero-surrogate:latest ${{ secrets.ECR_URI }}:latest
          docker push ${{ secrets.ECR_URI }}:latest

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster aero-surrogate-cluster \
            --service aero-surrogate-service \
            --force-new-deployment
```

---

## Repository Structure

```
transonic-aerodynamic-surrogate-modeling/
  app/
    main.py               # FastAPI app
    schemas.py            # Pydantic request/response models
  model/
    best_model.pkl        # Serialized best model
    X_scaler.pkl          # Feature scaler
    y_scaler.pkl          # Target scaler
    model_card.md         # Model performance summary
  notebooks/
    01_eda.ipynb
    02_baseline.ipynb
    03_random_forest.ipynb
    04_xgboost.ipynb
    05_dnn.ipynb
    06_evaluation.ipynb
  src/
    data.py
    evaluate.py
    train_dnn.py
  Dockerfile
  requirements.txt
  .github/
    workflows/
      deploy.yml
  README.md
  DEPLOYMENT.md           # This document
```

---

## Cost Estimate

| Resource | Spec | Monthly Cost |
|---|---|---|
| ECS Fargate | 0.25 vCPU, 0.5GB, 1 task | ~$8 |
| ECR storage | <1GB image | ~$0.10 |
| ALB | Low traffic | ~$18 |
| **Total** | | **~$26/month** |

Tear down when not in use to avoid charges. ECS service can be scaled to 0 tasks.

---

## Milestones

| Milestone | Description |
|---|---|
| M1 | Train and serialize best model, build FastAPI app, test locally |
| M2 | Dockerize, run container locally, validate `/predict` endpoint |
| M3 | Push image to ECR, deploy to ECS Fargate, verify public endpoint |
| M4 | Add GitHub Actions CI/CD pipeline |
| M5 | Load test endpoint, document latency vs direct model call overhead |
