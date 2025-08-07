# AI-Powered Transaction Fraud Detection System

A comprehensive fraud detection system using machine learning, graph neural networks, and real-time monitoring.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 8GB+ RAM recommended

### Installation & Setup

1. **Clone and navigate to project**
```bash
cd fraud/
```

2. **Build and run with Docker Compose**
```bash
# Production environment
docker-compose up -d

# Development environment with hot-reload
docker-compose -f docker-compose.dev.yml up -d
```

3. **Access the application**
- Main App: http://localhost:5050
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000 (admin/admin)

### Manual Setup (Alternative)

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Start services**
```bash
# Start MLflow
mlflow server --host 0.0.0.0 --port 5000 &

# Start application
python app.py
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Fraud Detection Application                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Flask     │  │   MLflow    │  │   Redis     │           │
│  │   API       │  │  Tracking   │  │   Cache     │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ PostgreSQL  │  │   Models    │  │   Reports   │           │
│  │   Redis     │  │   Files     │  │   Storage   │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables
```bash
# Core settings
FLASK_ENV=production
MLFLOW_TRACKING_URI=http://mlflow:5000
REDIS_URL=redis://redis:6379

# Database
POSTGRES_DB=fraud_detection
POSTGRES_USER=fraud_user
POSTGRES_PASSWORD=fraud_pass
```

### Docker Compose Profiles
```bash
# Production
docker-compose up -d

# Development with debugging
docker-compose -f docker-compose.dev.yml up -d

# With monitoring
docker-compose --profile monitoring up -d

# With Jupyter
docker-compose --profile dev up -d
```

## 📈 Monitoring

### Metrics Endpoints
- Application: http://localhost:5050/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Health Checks<|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|><|reserved_token_163839|>Plan for dockerizing the project:

Information Gathered:
- The project is a Flask-based fraud detection system with multiple ML models and background tasks.
- It uses MLflow for experiment tracking, Redis for caching, PostgreSQL for persistent storage, and has a frontend dashboard.
- There is an existing docker-compose.yml with mlflow and dashboard service but it is minimal.
- The main app runs on port 5050 and depends on mlflow.
- Requirements.txt is comprehensive with many dependencies.
- The app.py is the main entry point.
- There is no existing Dockerfile for the main app, so I created one with best practices including a non-root user, healthcheck, and proper environment setup.
- Created a .dockerignore to exclude unnecessary files.
- Extended docker-compose.yml to include services for the app, mlflow, redis, postgres, nginx reverse proxy, prometheus, grafana, and jupyter for dev.
- Created a docker-compose.dev.yml for development with debugging and hot reload.
- Created a Dockerfile.dev for development environment.
- Added nginx.conf for reverse proxy with rate limiting and security headers.
- Added prometheus.yml for monitoring.
- Added init.sql for initializing the postgres database with tables and views.

Plan:
- fraud/Dockerfile: Dockerfile for production app image.
- fraud/.dockerignore: To optimize docker build context.
- fraud/docker-compose.yml: Full production stack including app, mlflow, redis, postgres, nginx, prometheus, grafana, jupyter.
- fraud/docker-compose.dev.yml: Development stack with debugging and hot reload.
- fraud/Dockerfile.dev: Development Dockerfile with debugging tools.
- fraud/nginx.conf: Nginx reverse proxy config.
- fraud/prometheus.yml: Prometheus config for monitoring.
- fraud/init.sql: Postgres DB initialization script.

Dependent Files to be edited:
- None, all new files created.

Followup steps:
- Build and run the docker-compose stack.
- Verify the app runs correctly and connects to mlflow, redis, postgres.
- Verify nginx reverse proxy works.
- Verify monitoring with prometheus and grafana.
- Test development environment with docker-compose.dev.yml.
- Adjust configurations as needed based on testing.

Please confirm if I can proceed with any additional dockerization tasks or if you want me to help you run and test the docker setup.

<ask_followup_question>
<question>Do you confirm this dockerization plan and want me to assist with running or testing the docker containers?</question>
</ask_followup_question>
