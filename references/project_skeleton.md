telecom-customer-churn
├── LICENSE.txt                <- Project's license (MIT)
├── README.md                  <- The top-level README for developers using this project
├── .env                       <- Environment variables (never committed)
├── .gitignore                 <- Files to ignore by Git
├── .dockerignore              <- Files to ignore by Docker
├── dvc.yaml                   <- The Pipeline Conductor (DVC stages)
├── pyproject.toml             <- UV dependency definitions & tool config (ruff, mypy, pytest)
├── main.py                    <- Pipeline Orchestrator (Script mode)
├── Dockerfile                 <- Production container definition
├── GEMINI.md                  <- Antigravity global rules for the AI assistant
│
├── .github/
│   └── workflows/             <- CI/CD workflows (GitHub Actions)
│
├── artifacts/                 <- Generated artifacts (models, metrics, transformed data)
│   ├── gx/                    <- Great Expectations validation results
│   └── models/                <- Serialized model artifacts
│
├── config/                    <- Centralize all configuration files ("source of truth")
│   ├── config.yaml            <- System paths (artifacts/data) — immutable structure
│   ├── params.yaml            <- Hyperparameters (tunable values) & MLflow config
│   └── schema.yaml            <- Data schema definitions (column types)
│
├── data/
│   ├── external/              <- Data from third party sources
│   └── raw/                   <- The original, immutable data dump
│       ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  <- Primary Telco dataset (7,043 rows)
│       └── customer_churn.csv                     <- Reference dataset with ticket_notes (100 rows)
│
├── logs/                      <- Logs of the pipeline execution (rotating file handler)
│
├── notebooks/                 <- Jupyter notebooks (EDA, prototyping)
│
├── references/                <- Data dictionaries, manuals, and all other explanatory materials
│   └── project_skeleton.md    <- This file
│
├── reports/                   <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── docs/                  <- Generated documents to be used in reporting
│   │   └── ticket_notes_enrichment.md  <- Strategy for synthetic note generation
│   └── figures/               <- Generated graphics and figures to be used in reporting
│
├── tests/                     <- Unit tests and integration tests (pytest)
│
├── ui/                        <- User Interface (Gradio app)
│
└── src/                            <- Source code for use in this project
    │
    ├── __init__.py                 <- Makes src a Python module
    │
    ├── api/                        <- FastAPI Serving Layer (Inference Pipeline)
    │   └── __init__.py
    │
    ├── components/                 <- Business Logic / Workers (The "How")
    │   └── __init__.py             <- Data ingestion, validation, transformation, training, evaluation
    │
    ├── config/                     <- Configuration Management (The "Brain")
    │   ├── __init__.py
    │   └── configuration.py        <- ConfigurationManager: reads YAMLs, hydrates dataclass configs
    │
    ├── constants/                  <- Centralize constants for the project
    │   └── __init__.py             <- Path constants (CONFIG_FILE_PATH, PARAMS_FILE_PATH, etc.)
    │
    ├── enrichment/                 <- AI-Powered Data Enrichment (Agentic Pipeline)
    │   └── prompts/                <- Versioned prompt templates for synthetic note generation
    │
    ├── entity/                     <- Data entities & contracts
    │   ├── __init__.py
    │   └── config_entity.py        <- Frozen dataclasses for pipeline configs + Pydantic row validators
    │
    ├── pipeline/                   <- Execution Stages (The "Conductor")
    │   └── __init__.py
    │
    └── utils/                      <- Common utilities
        ├── __init__.py
        ├── common.py               <- YAML/JSON readers, directory creation, file size helpers
        ├── exception.py            <- Custom Error Handling (CustomException with traceback)
        ├── logger.py               <- Centralized logging (RotatingFileHandler + RichHandler)
        └── mlflow_config.py        <- MLflow URI resolution (env-aware: local/staging/production)
