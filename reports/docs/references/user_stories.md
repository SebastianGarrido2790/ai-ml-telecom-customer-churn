# User Stories & Problem Framing: Telecom Customer Churn

## 1. Problem Framing (The "Why")

### The Business Context
In the highly competitive telecommunications industry, customer acquisition costs are 5x to 25x higher than retention costs. Predicting **churn** (when a customer cancels their service) is a critical business priority for maintaining market share and revenue stability.

### The Gap
Traditional churn models rely primarily on **"Hard Signals"**:
- Billing cycles and payment methods.
- Usage patterns (streaming, data caps).
- Contract types and tenure.

However, these models often ignore **"Soft Signals"** (Human Sentiment):
- Unresolved technical complaints.
- Frustration with billing discrepancies expressed in support tickets.
- Perceived lack of value in specific tiers.

**The Problem:** Existing datasets often lack these qualitative "Soft Signals," leading to models that miss the nuanced emotional triggers of churn.

### Our Hypothesis
By using **Agentic AI** to synthesize high-fidelity qualitative data (ticket notes) based on existing customer features, and then processing that data through an **NLP-aware MLOps pipeline**, we can achieve higher predictive accuracy and provide stakeholders with a clear "why" behind every churn risk score.

---

## 2. User Stories

### Story 1: The Customer Retention Manager
> **As a** Retention Manager,  
> **I want to** access a dashboard showing a prioritized list of at-risk customers along with a summarized "interaction sentiment,"  
> **so that** I can proactively offer personalized retention packages that address the specific reason they are frustrated.

### Story 2: The Data Scientist
> **As a** Data Scientist,  
> **I want** an automated, versioned pipeline that produces "AI-enriched" features,  
> **so that** I can experiment with different NLP models and traditional ML algorithms while tracking results in a centralized registry (MLflow).

### Story 3: The MLOps Engineer
> **As an** MLOps Engineer,  
> **I want** to deploy the inference system as a containerized microservice with automated data validation contracts (GX),  
> **so that** the production system fails loudly on schema drift and scales independently to meet API demand.

### Story 4: The Support Lead
> **As a** Support Lead,  
> **I want** real-time churn propensity scores integrated via API when a customer calls in,  
> **so that** my agents can adjust their tone and priority levels for customers who are already at a high risk of leaving.

---

## 3. Success Definition
- **Primary KPI:** Reduction in false negatives (missing customers who are about to churn).
- **Secondary KPI:** Qualitative "Ticket Notes" provide measurable feature importance in the final model (XGBoost/LightGBM).
- **Technical KPI:** Deployment consistency (>95% automated test coverage for the FTI pipeline).
