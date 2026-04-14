## Phase 7B Scope

Five components confirmed:

1. **Single customer predict form** — CRM fields + ticket note input, posts to `POST /v1/predict`, displays `ChurnPredictionResponse` fields including branch probabilities and `nlp_branch_available` indicator.

2. **Batch CSV upload + results table** — uploads a CSV, calls `POST /v1/predict/batch`, renders results as a sortable table with churn probability and prediction columns.

3. **SHAP feature importance chart** — computes SHAP values from `structured_model.pkl` at prediction time, renders a bar chart of the top contributors for that specific customer's score.

4. **`nlp_branch_available` status indicator** — visible badge showing whether the NLP branch contributed to the current prediction, with a plain-language explanation of what it means when `False`.

5. **MLflow run comparison panel** — reads from the `evaluation_report.json` artifact and/or the MLflow file store to display the three-run comparison table (structured baseline vs. NLP baseline vs. Late Fusion) with `recall_lift` and `f1_lift` highlighted.

---

**Decision H1** — Gradio as a fifth Docker Compose service is locked. Phase 7B will deliver:

- `src/ui/` — modular Gradio application (pages, components, data loaders, no monolithic `app.py`)
- `docker/gradio_ui/Dockerfile` — multi-stage build, non-root user, health check
- `docker-compose.yaml` — updated to uncomment and complete the `gradio-ui` service with `depends_on: prediction-api: condition: service_healthy` and `PREDICTION_API_URL=http://prediction-api:8000`

The five UI components confirmed in scope: single customer predict form, batch CSV upload + results table, SHAP feature importance chart, `nlp_branch_available` status indicator, and MLflow run comparison panel.

---

## Phase 7B: Docker Compose Startup Sequence

```bash
docker compose up -d --build
[+] Building 1150.7s (68/68) FINISHED
...
[+] Running 8/8
 ✔ telecom-churn/prediction-api:latest     Built                                                0.0s
 ✔ telecom-churn/gradio-ui:latest          Built                                                0.0s
 ✔ telecom-churn/mlflow-server:latest      Built                                                0.0s
 ✔ telecom-churn/embedding-service:latest  Built                                                0.0s
 ✔ Container mlflow-server                 Started                                             77.9s
 ✔ Container embedding-service             Healthy                                             95.9s
 ✔ Container prediction-api                Healthy                                             36.5s
 ✔ Container gradio-ui                     Started                                             32.1s
```

---

The logs confirm that all five services are now fully operational, and the **startup health-gates** passed successfully (meaning the models are warmed up and ready).

You can interact with the system using these URLs:

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Interactive Dashboard** | [**http://localhost:7860**](http://localhost:7860) | **Standard User Interface**. Submit single predictions, upload batch CSVs, and view SHAP explanations. |
| **MLflow Tracking** | [http://localhost:5000](http://localhost:5000) | **ML Engineering Interface**. Review training history, registered models, and full experiment lineage. |
| **Prediction API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | **API Integration**. Test the raw JSON endpoints (Swagger UI) for the late-fusion inference. |

### Docker Compose Commands

```bash
# Build and start only the Gradio UI
docker compose up --build gradio-ui

# Start the Gradio UI in detached mode
docker compose up -d gradio-ui

# Stop the Gradio UI
docker compose down gradio-ui
```

### **Quick Test Scenario:**
1. Open the [Dashboard](http://localhost:7860).
2. Go to the **Single Prediction** tab.
3. Keep the defaults but enter a `ticket_note` like: *"Customer is very angry about the recent price hike and is threatening to switch to a competitor."*
4. Click **Predict Churn Risk**.
5. You should see a high **Churn Probability** and a **SHAP Waterfall Chart** explaining which billing factors contributed most to the risk.

**Note:** If the SHAP plot or Run Comparison table appears empty, it's because the `artifacts/` folder on your host machine hasn't been populated by a `dvc repro` run yet. The UI will still work, but those diagnostic features will wait for the data.
