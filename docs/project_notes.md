# 1. Project overview

This project is a synthetic HR analytics and nepotism-risk analysis pipeline. Its current objective is to generate simulated company candidate and employee data under multiple HR regimes and use that data to estimate hiring favoritism, promotion favoritism, concentration of connected employees, and suspicious merit-outcome mismatches.

The current implemented pipeline is a 4-model workflow:

- Model 1: Hiring model. Predicts or explains which candidates are hired using merit variables, connection variables, and discretionary channel features.
- Model 2: Promotion model. Predicts or explains which employees are promoted using performance, tenure, role/salary context, and connection variables.
- Model 3: Network nepotism model. Produces manager-level and department-level nepotism risk scores from concentration and connectedness patterns in the employee population.
- Model 4: Anomaly / merit-outcome mismatch model. In the current codebase this is no longer only generic anomaly detection. It currently scores suspicious hires and suspicious promotions based on low merit or low performance combined with connection intensity and discretionary channels.

The longer-term simulation concept is documented separately in `docs/simulation.txt`, but the current repository is still centered on the data-generation and analysis pipeline rather than a time-based simulation engine or end-user application.

## Recent improvements (UI/UX Polish Pass)

The web application received a product-level UI/UX polish to prepare it for professional interview/demo walkthroughs:

- **Visual design**: Enhanced CSS styling and consistent layout patterns across all pages
- **Chart improvements**: Improved chart formatting and consistency across Data Summary, Risk Dashboard, Statistical Analysis, and Organizational Impact views in the current web app
- **Text/explanation upgrades**: Upgraded copy throughout from demo language to professional analytics language; added clearer section introductions and interpretation guidance
- **Data Summary methodology**: Added a collapsible "Research & Methodology Basis" section explaining the research grounding (favoritism patterns, nepotism proxies, patronage literature, systematic reviews)
- **Risk Dashboard**: Improved Model 3/4 visual hierarchy with labeled containers and better metric card organization
- **Statistical Analysis**: Enhanced sensitivity curve explanations and matched-pair interpretation text
- **Organizational Impact**: Improved executive comparison cards and section explanations with proxy usage clearly noted

# 2. What currently exists

The items below reflect the current code and files that are actually present in the repository.

- `src/generate_data.py`
  - Purpose: generates synthetic candidate and employee data across three scenarios: Merit-based, Moderate favoritism, and High nepotism risk.
  - Status: implemented.
  - Notes: exports `data/generated/nepotism_synthetic_data.xlsx`; includes calibration logic so scenario outputs hit target hire/promotion behavior and differing connection dominance.

- `data/generated/nepotism_synthetic_data.xlsx`
  - Purpose: master generated dataset exported from the data generation step.
  - Status: implemented/output exists.
  - Notes: serves as the input to the processing pipeline.

- `src/prepare_model_data.py`
  - Purpose: loads the generated Excel file, validates structure and values, cleans dtypes, engineers candidate and employee features, and creates manager/department aggregates.
  - Status: implemented.
  - Notes: writes model-ready CSVs into `data/processed/`.

- `data/processed/candidates_model.csv`
  - Purpose: model-ready candidate dataset for hiring analysis.
  - Status: implemented/output exists.

- `data/processed/employees_model.csv`
  - Purpose: model-ready employee dataset for promotion, network, and anomaly analysis.
  - Status: implemented/output exists.

- `data/processed/manager_features.csv`
  - Purpose: aggregated manager-level features derived from employee records.
  - Status: implemented/output exists.

- `data/processed/department_features.csv`
  - Purpose: aggregated department-level features derived from employee records.
  - Status: implemented/output exists.

- `src/model_hiring.py`
  - Purpose: fits multiple logistic-regression hiring models, including explanatory, compact, weighted, and calibrated variants.
  - Status: implemented.
  - Notes: also fits scenario-specific explanatory models when enough rows exist.

- `outputs/hiring_model/`
  - Purpose: saved hiring-model outputs.
  - Status: implemented/output exists.
  - Contents currently present:
  - `hiring_metrics.csv`
  - `hiring_coefficients.csv`
  - `hiring_explanatory_inference.csv`
  - `candidate_predictions.csv`
  - `roc_curve.png`
  - `pr_curve.png`

- `src/model_promotion.py`
  - Purpose: fits multiple logistic-regression promotion models using employee-level features.
  - Status: implemented.
  - Notes: mirrors the hiring-model structure with explanatory, compact, weighted, calibrated, and scenario-specific variants.

- `outputs/promotion_model/`
  - Purpose: saved promotion-model outputs.
  - Status: implemented/output exists.
  - Contents currently present:
  - `promotion_metrics.csv`
  - `promotion_coefficients.csv`
  - `promotion_explanatory_inference.csv`
  - `employee_predictions.csv`
  - `roc_curve.png`
  - `pr_curve.png`

- `src/model_network_nepotism.py`
  - Purpose: computes manager-level and department-level nepotism risk scores from concentration patterns such as family links, school/city clustering, surname concentration, connected employee share, and connected promotion share.
  - Status: implemented.
  - Notes: this is a structured risk-scoring model, not a full network graph application.

- `outputs/network_model/`
  - Purpose: saved network-risk outputs and charts.
  - Status: implemented/output exists.
  - Contents currently present:
  - `manager_nepotism_scores.csv`
  - `department_nepotism_scores.csv`
  - `top_risky_managers.csv`
  - `top_risky_departments.csv`
  - `top_risky_managers.png`
  - `top_risky_departments.png`

- `src/model_anomaly.py`
  - Purpose: scores suspicious hires and suspicious promotions using a merit/performance mismatch logic combined with connection indicators and discretionary channels.
  - Status: implemented.
  - Notes: this is more specific than a generic anomaly detector. It currently outputs hiring and promotion anomaly results only.

- `outputs/anomaly_model/`
  - Purpose: saved anomaly and suspicious-case outputs.
  - Status: partially consistent.
  - Contents currently present and clearly generated by the current script:
  - `hiring_anomaly_scores.csv`
  - `promotion_anomaly_scores.csv`
  - `top_hiring_anomalies.csv`
  - `top_promotion_anomalies.csv`
  - `hiring_anomaly_rate_by_scenario.csv`
  - `promotion_anomaly_rate_by_scenario.csv`
  - `hiring_anomaly_scores.png`
  - `promotion_anomaly_scores.png`
  - Additional files currently present but not produced by the current `src/model_anomaly.py` implementation:
  - `manager_anomaly_scores.csv`
  - `department_anomaly_scores.csv`
  - `top_manager_anomalies.csv`
  - `top_department_anomalies.csv`
  - `manager_anomaly_scores.png`
  - `department_anomaly_scores.png`
  - Interpretation: these appear to be legacy outputs from an earlier version or a removed script path. They should not be treated as fully current without checking their generation source.

- `docs/project_notes.md`
  - Purpose: state-of-project notes documenting the current implemented pipeline and gaps.
  - Status: implemented and current.

- `docs/simulation.txt`
  - Purpose: describes the planned future simulation engine and scenario comparison logic.
  - Status: planned/design documentation, not implemented as code in the current repo.
  - Notes: includes the intended 24-month workforce simulation, review cycles, quitting/replacement logic, and dashboard-style outcome comparisons.

- `requirements.txt`
  - Purpose: Python dependencies for the project.
  - Status: implemented.
  - Notes: includes `pandas`, `numpy`, `openpyxl`, `scikit-learn`, `matplotlib`, `networkx`, `pyyaml`, `faker`, `statsmodels`, `fastapi`, and `uvicorn`.

- `backend/api.py`
  - Purpose: FastAPI backend for the web-migration path.
  - Status: implemented, partial migration.
  - Notes: currently exposes home metadata, Data Summary data, Risk Dashboard data, Statistical Analysis data, Organizational Impact data, hiring/promotion reference-data endpoints, and live hiring/promotion prediction endpoints that reuse the existing Python predictor logic.

- `frontend/`
  - Purpose: static web frontend served by the FastAPI backend.
  - Status: implemented, partial migration.
  - Notes: currently includes Home, Hiring Predictor, Promotion Predictor, Data Summary, Risk Dashboard, Statistical Analysis, and Organizational Impact views.

- `app_utils/`
  - Purpose: helper layer for live prediction inside the app.
  - Status: implemented.
  - Notes: includes model loading, single-row input preprocessing, probability/driver helpers, data-summary aggregation helpers, dashboard-output loading helpers, statistical-analysis helpers for bulk scoring, pair matching, and sensitivity curves, and organizational-impact helpers for scenario metrics and statsmodels explanatory inference tables.

- `artifacts/predictor_models/`
  - Purpose: persisted predictor bundles for Model 1 and Model 2.
  - Status: implemented/output exists.
  - Notes: currently contains serialized hiring and promotion predictor bundles used by the app to avoid live refitting during normal use.

- UI/app-related files
  - Status: implemented.
  - Notes: the supported local interface is now the FastAPI + static-frontend web app, covering Home, Hiring Predictor, Promotion Predictor, Data Summary, Risk Dashboard, Statistical Analysis, and Organizational Impact.

- Orchestration / packaging files
  - Status: partial.
  - Notes: the project is runnable as individual scripts and through the web-app launcher (`Run_Nepotism_Web_App.bat`), but there is still no README or broader packaging layer beyond the existing persisted app predictor bundles under `artifacts/predictor_models/`.

# 3. What is validated / already in good shape

The following statements are supported by the code and currently saved outputs.

- The synthetic data generation pipeline appears implemented and usable.
  - `src/generate_data.py` exists and exports the generated Excel dataset used by downstream steps.
  - The current processed datasets and model outputs indicate that the generation step has already been run successfully.

- The processed data pipeline appears implemented and in good shape.
  - `src/prepare_model_data.py` performs validation, feature engineering, and manager/department aggregation.
  - The processed CSVs exist in `data/processed/`.

- The hiring model appears relatively strong and usable.
  - The script exists, multiple model variants are implemented, and metrics/coefficients/predictions/plots are being generated.
  - Current `outputs/hiring_model/hiring_metrics.csv` shows ROC-AUC values around the high-0.8 range for the stronger variants.

- The promotion model appears implemented and usable, but weaker than the hiring model.
  - The script and outputs exist and follow the same structure as the hiring model.
  - Current promotion metrics are materially lower and some variants show poor recall at the default threshold, so this model should be treated as usable but less mature.

- The network nepotism model exists and produces manager/department risk outputs.
  - `src/model_network_nepotism.py` is implemented.
  - CSV rankings and chart outputs are already present in `outputs/network_model/`.

- The anomaly / suspicious-outcome model exists and is already producing scenario-level outputs.
  - Current outputs include suspicious hires, suspicious promotions, top suspicious cases, scenario-level anomaly rates, and charts.
  - Based on the present code, this model has already moved beyond generic anomaly detection into a more explicit low-merit/low-performance plus high-connection scoring approach.

- The outputs structure is already useful for continuation.
  - Each model stage writes CSV outputs and charts into a dedicated folder.
  - This makes the current project state reproducible at the artifact level, and the app now also persists hiring/promotion predictor bundles under `artifacts/predictor_models/`.

# 4. What is still planned / not finished

The following gaps or pending components are real gaps in the current repository.

- The 24-month simulation engine described in `docs/simulation.txt` is not implemented.
  - No script was found for monthly production simulation, quarterly review cycles, quitting logic, replacement hiring, or scenario-by-scenario time progression.

- The current input tool is limited to local single-record prediction for Model 1 and Model 2.
  - It does not yet support batch upload, export workflows, or direct access to Model 3 and Model 4 outputs.

- The current data summary screen is descriptive rather than exploratory.
  - It summarizes generated and processed data with metrics, scenario comparisons, charts, and scenario-difference tables, but it does not yet provide deeper filtering, downloads, or raw-record exploration.

- The current Risk Dashboard depends on saved output files.
  - It reads the current CSV and PNG artifacts from `outputs/network_model/` and `outputs/anomaly_model/`, so it will only show sections that have already been generated.

- The current Statistical Analysis page depends on the processed datasets and the persisted default predictor bundles.
  - It uses real processed records and real model predictions for matched-pair comparisons and sensitivity curves, uses pair selectors so one strong matched comparison is shown at a time rather than listing every matched pair at once, and now restricts matched pairs to the same scenario for cleaner within-regime interpretation.

- The current Organizational Impact page is comparative and output-driven rather than fully interactive.
  - It computes scenario-level quality, fairness, efficiency, and structural-risk comparisons from the current processed data and output CSVs, and it uses `performance_score` as the production proxy because no direct production field currently exists in the employee dataset.

- The backend/frontend migration now covers the current local inspection surface.
  - The FastAPI backend serves a static frontend and reuses the existing predictor-bundle loading, preprocessing, driver-summary, logistic-math logic, data-summary helper logic, dashboard-output helper logic, statistical-analysis helper logic, and organizational-impact helper logic.
  - The web frontend currently covers Home, Hiring Predictor, Promotion Predictor, Data Summary, Risk Dashboard, Statistical Analysis, and Organizational Impact.

- Predictor artifact persistence now exists, but it is app-focused rather than a full training-pipeline packaging layer.
  - The web app now loads persisted hiring/promotion predictor bundles from `artifacts/predictor_models/`.
  - The standalone model scripts still primarily export metrics, predictions, coefficients, and charts rather than a broader artifact registry for every downstream use case.

- Explanatory p-value inference is now available, but it is parallel to the predictor layer rather than replacing it.
  - The app now generates and reads `outputs/hiring_model/hiring_explanatory_inference.csv` and `outputs/promotion_model/promotion_explanatory_inference.csv` from statsmodels Logit fits for coefficient inference.
  - Live prediction in the app still uses the sklearn-based predictor bundles rather than the statsmodels explanatory fits.

- Repository-level documentation is incomplete.
  - The docs folder is now more current, but there is still no README or end-to-end run guide in the project root.

- The anomaly output folder has a current-vs-legacy mismatch.
  - Manager and department anomaly files exist in `outputs/anomaly_model/`, but the current `src/model_anomaly.py` does not generate them.
  - This should be cleaned up or documented before presenting the repository as a fully consistent product.

# 5. What should be built next

Recommended next build order, based on the current repository state and current product direction:

- 1. Batch input and export workflow for Model 1 and Model 2
  - What it is: a controlled upload/download workflow for scoring multiple candidate or employee rows at once.
  - Why it is next: the local single-record predictor is useful for demos, but practical analyst usage will need repeatable multi-row scoring.
  - What it should reuse: `app_utils/model_loader.py`, `app_utils/input_preprocessing.py`, and the existing processed-data schemas.

- 2. Expanded data exploration controls
  - What it is: richer filters, export options, and optional distribution views on top of the current Data Summary page.
  - Why it is next: the current summary page is strong for demos, but still intentionally high level.
  - What it should reuse: `frontend/app.js`, `backend/api.py`, `app_utils/data_summary_helpers.py`, and the processed CSV datasets.

- 3. Expanded dashboard controls for Model 3 and Model 4
  - What it is: deeper filters, trend views, and cleaner handling of historical vs current anomaly artifacts in the Risk Dashboard.
  - Why it is next: the dashboard now exists, but it is still driven by saved output files rather than interactive recomputation.
  - What it should reuse: `frontend/app.js`, `backend/api.py`, `app_utils/dashboard_helpers.py`, and the current output folders.

After those three steps, the next major product step would be the full simulation engine described in `docs/simulation.txt`, assuming the project direction remains aligned with that document.

# 6. Implementation notes / architecture notes

The project is currently organized as a Python analytics pipeline with separate scripts for each major stage.

- Main data folders
  - `data/generated/`: generated master Excel dataset.
  - `data/processed/`: model-ready CSV datasets and manager/department aggregates.

- Main model scripts
  - `src/generate_data.py`: synthetic data generation and scenario calibration.
  - `src/prepare_model_data.py`: validation, feature engineering, and processed dataset creation.
  - `src/model_hiring.py`: hiring-model training, evaluation, coefficients, predictions, and plots.
  - `src/model_promotion.py`: promotion-model training, evaluation, coefficients, predictions, and plots.
  - `src/model_network_nepotism.py`: manager/department risk-score generation and rankings.
  - `src/model_anomaly.py`: suspicious hire and suspicious promotion scoring.

- Output folders
  - `outputs/hiring_model/`: hiring metrics, coefficients, predictions, ROC/PR plots.
  - `outputs/promotion_model/`: promotion metrics, coefficients, predictions, ROC/PR plots.
  - `outputs/network_model/`: manager/department risk scores, top-ranked entities, charts.
  - `outputs/anomaly_model/`: suspicious-case outputs and anomaly-rate summaries; contains some legacy files that should be treated carefully.

- Documentation
  - `docs/project_notes.md`: current state-of-project notes.
  - `docs/simulation.txt`: planned future simulation design; useful for roadmap, not current implementation.

- UI/app location
  - `backend/api.py`: FastAPI backend for the web-migration path.
  - `frontend/`: static frontend served by FastAPI.
  - `app_utils/`: persisted model loading, single-row preprocessing, prediction helper logic, cached data-summary aggregation logic, cached dashboard-output loading logic, statistical-analysis helpers, and organizational-impact helpers.
  - `artifacts/predictor_models/`: persisted hiring/promotion predictor bundles used by the app.

- Important continuation notes
  - The project is still primarily a script-based analytics pipeline, now wrapped by a local FastAPI + static-frontend web app for prediction, dataset inspection, and risk-output review.
  - The Data Summary page reads directly from the generated workbook and processed CSV files and skips only the affected sections if a file or required column is missing.
  - The Risk Dashboard reads current outputs from `outputs/network_model/` and `outputs/anomaly_model/`; it currently shows Model 3 manager/department risk outputs and Model 4 suspicious hires/promotions, while older manager/department anomaly files are treated as legacy artifacts.
  - The FastAPI backend and static frontend expose the Data Summary, Risk Dashboard, Statistical Analysis, and Organizational Impact views by serializing the same processed-data, output-artifact, matched-pair/sensitivity, and impact/inference helper bundles used by the analytics layer.
  - The shared model-loader, Data Summary, Risk Dashboard, Statistical Analysis, and Organizational Impact helper layers use framework-neutral Python caching so the backend can import them cleanly.
  - The Statistical Analysis page uses real processed data and the current persisted predictor bundles to show same-scenario matched candidate/employee comparisons and one-variable-at-a-time sensitivity curves, with pair selection controls to keep the page compact.
  - The Organizational Impact page combines processed data, Model 3 outputs, Model 4 anomaly-rate outputs, and statsmodels explanatory inference tables to compare scenario-level consequences.
  - The FastAPI backend and static frontend reuse the same Python predictor layer rather than duplicating model logic in JavaScript.
  - The predictor pages now prefer persisted model bundles from `artifacts/predictor_models/` and rebuild them only when the processed input data, metrics file, or model module changes.
  - App startup is lighter than before because predictor model modules are imported lazily and heavy summary/dashboard aggregates are precomputed inside cached bundle loaders.
  - The explanatory coefficient tables with p-values come from statsmodels Logit fits over the explanatory feature sets; they are saved under `outputs/hiring_model/` and `outputs/promotion_model/`, while live app prediction still uses sklearn-based models.
  - Future work should preserve the existing folder structure and reuse the processed datasets and output folders instead of creating parallel pipelines.
  - If a UI is added next, it should be treated as a thin layer on top of the existing scripts and outputs rather than a rewrite of the core modeling logic.
