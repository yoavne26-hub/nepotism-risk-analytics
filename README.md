# Nepotism Risk Analytics

Nepotism Risk Analytics is a synthetic HR analytics platform for modeling how favoritism and connection-based decision-making affect hiring quality, promotion fairness, structural risk, suspicious decisions, and long-run organizational outcomes.

The repository combines a Python analytics pipeline with a local web application built on FastAPI and a static frontend. It is designed for portfolio presentation, academic demonstration, and analytical prototyping around workforce fairness, risk detection, and scenario comparison.

## Product Highlights

- Multi-model analytics pipeline covering hiring, promotion, structural nepotism risk, and suspicious decision detection
- Synthetic workforce datasets generated across three HR regimes:
  - Merit-based
  - Moderate favoritism
  - High nepotism risk
- Local web application for:
  - Hiring probability prediction
  - Promotion probability prediction
  - Data summary and scenario comparisons
  - Risk dashboard for manager and department exposure
  - Statistical analysis and sensitivity exploration
  - Organizational impact comparison across scenarios
- Persisted predictor bundles for fast app startup without retraining during normal use
- Exported metrics, ranked outputs, and plots for downstream reporting

## Why This Project Matters

Most analytics work around favoritism stops at descriptive reporting. This project goes further by creating an end-to-end analytical environment that can:

- simulate alternative HR regimes
- quantify how connected candidates and employees alter outcomes
- estimate structural nepotism concentration at manager and department level
- surface suspicious low-merit or low-performance decisions
- compare the organizational cost of favoritism through fairness, efficiency, and risk outputs

The result is a product-style analytical system rather than a single notebook or isolated model.

## Product Gallery

Add screenshots to `docs/screenshots/` using the file names below. The README is already structured so the gallery can be completed without changing the rest of the document.

| View | Suggested File | Purpose |
| --- | --- | --- |
| Dashboard | `docs/screenshots/dashboard-overview.png` | High-level KPI cards and product landing page |
| Hiring Predictor | `docs/screenshots/hiring-predictor.png` | Candidate scoring workflow |
| Promotion Predictor | `docs/screenshots/promotion-predictor.png` | Employee promotion scoring workflow |
| Data Summary | `docs/screenshots/data-summary.png` | Scenario metrics and dataset overview |
| Risk Dashboard | `docs/screenshots/risk-dashboard.png` | Network and anomaly outputs |
| Statistical Analysis | `docs/screenshots/statistical-analysis.png` | Sensitivity charts and matched-pair analysis |
| Organizational Impact | `docs/screenshots/organizational-impact.png` | Cross-scenario fairness and efficiency comparisons |

## Core Capabilities

### 1. Hiring Model

Predicts hiring probability from merit indicators, connection indicators, and discretionary recruitment channels.

Typical inputs include:

- education level
- GPA and test/interview scores
- experience
- referral and family-link signals
- connection strength
- discretionary channel indicators

### 2. Promotion Model

Predicts promotion probability from performance, tenure, role context, salary, experience, and connection-related features.

### 3. Network Nepotism Risk Model

Produces manager-level and department-level risk scores from concentration and connectedness patterns in the employee population.

### 4. Suspicious Decision / Anomaly Model

Flags suspicious hires and promotions where low merit or low performance appears alongside strong connection indicators or discretionary routing.

## Scenario Design

The current analytics pipeline compares three organizational regimes:

- **Merit-based**: merit dominates hiring and promotion decisions
- **Moderate favoritism**: merit still matters, but connected profiles gain partial advantage
- **High nepotism risk**: connection and relationship signals exert a strong decision influence

These scenarios allow the project to compare fairness, efficiency, and structural exposure under different governance assumptions.

## Application Modules

The local web application currently exposes the following product surfaces:

- **Dashboard**: KPI overview and cross-model orientation
- **Hiring Predictor**: single-record candidate probability scoring
- **Promotion Predictor**: single-record employee promotion scoring
- **Data Summary**: dataset dimensions, scenario metrics, and scenario comparison tables
- **Risk Dashboard**: top risky managers, top risky departments, suspicious hires, and suspicious promotions
- **Statistical Analysis**: matched-pair comparisons and one-variable-at-a-time sensitivity curves
- **Organizational Impact**: scenario-level comparison of hiring quality, promotion fairness, efficiency, and structural risk

## Repository Structure

```text
.
|-- app_utils/                 Helper modules used by the web application
|-- artifacts/
|   `-- predictor_models/      Persisted hiring and promotion bundles
|-- backend/                   FastAPI backend
|-- data/
|   |-- generated/             Synthetic source workbook
|   `-- processed/             Model-ready CSV datasets
|-- docs/                      Project notes, roadmap, and screenshot placeholders
|-- frontend/                  Static frontend served by FastAPI
|-- outputs/
|   |-- anomaly_model/         Suspicious-case outputs and scenario anomaly summaries
|   |-- hiring_model/          Hiring metrics, predictions, coefficients, and plots
|   |-- network_model/         Manager/department risk outputs
|   `-- promotion_model/       Promotion metrics, predictions, coefficients, and plots
|-- src/                       Data generation and modeling pipeline
|-- requirements.txt
`-- Run_Nepotism_Web_App.bat
```

## Technology Stack

- Python
- FastAPI
- Uvicorn
- Pandas
- NumPy
- scikit-learn
- statsmodels
- matplotlib
- openpyxl
- networkx
- Faker
- Vanilla JavaScript and CSS frontend

## Getting Started

### Prerequisites

- Python 3.11+ recommended
- Windows environment for the included launcher script

### Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run The Web Application

Option 1:

```powershell
.\Run_Nepotism_Web_App.bat
```

Option 2:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Data And Modeling Workflow

The project currently follows this workflow:

1. Generate synthetic candidate and employee data under multiple HR regimes
2. Prepare model-ready candidate, employee, manager, and department datasets
3. Train and evaluate hiring and promotion models
4. Compute network nepotism risk scores
5. Score suspicious hiring and promotion outcomes
6. Serve the analytical surfaces through the local FastAPI web application

## Analytical Outputs

The repository already includes generated outputs for demonstration and inspection, including:

- hiring and promotion metrics
- explanatory coefficients and inference tables
- candidate and employee prediction outputs
- ROC and precision-recall charts
- ranked risky managers and departments
- suspicious hires and suspicious promotions
- scenario-level anomaly summaries

## Professional Use Cases

This project is well suited for:

- analytics engineering portfolios
- operations research and industrial engineering demonstrations
- HR analytics capstone presentations
- model interpretability and fairness case studies
- governance, compliance, or audit prototyping discussions

## Roadmap

Planned next steps include:

- batch upload and export workflows for prediction pages
- deeper filtering and exploration in the web interface
- expanded dashboard controls for risk and anomaly analysis
- a full multi-period workforce simulation engine based on the design documented in `docs/simulation.txt`

## Important Notes

- The data is synthetic and intended for analytical experimentation and product demonstration.
- The current app is a local product surface, not a hosted SaaS deployment.
- Some legacy anomaly artifacts remain in `outputs/anomaly_model/`; the current analytics workflow should be interpreted using the files generated by the active scripts and helpers.

## Author

**Yoav Neman**  
Industrial Engineering and Analytics Project

## License

This repository does not currently include a license file. Add one before public reuse or commercial distribution if you want to define usage terms explicitly.
