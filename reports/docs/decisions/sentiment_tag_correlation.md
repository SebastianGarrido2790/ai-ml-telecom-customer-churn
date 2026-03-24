## Sentiment Tag Correlation

The dataset analysis reveals a critical design decision point before planning begins.

The sentiment-churn cross-tabulation tells a decisive story:

| Sentiment | Churn=No | Churn=Yes |
|---|---|---|
| Frustrated | 13 | **1,838** |
| Satisfied | 4,815 | 14 |
| Neutral | 242 | 2 |

The AI agent in Phase 2 generated sentiment tags that are **near-deterministically correlated with the target variable.** `Frustrated` predicts `Churn=Yes` with 99.3% accuracy on its own. This is a **data leakage risk** that must be addressed before designing the training pipeline. It does not invalidate Phase 2 — the ticket notes themselves are legitimate enrichment — but it does affect how `primary_sentiment_tag` is handled across the two branches.

---

### Decision 1 — `primary_sentiment_tag` Assignment

The sentiment tag was derived by the LLM using the `Churn` label as context (from the `CustomerInputContext` input contract in Phase 2). It is therefore a synthetic signal, not an observable one. In production, a real customer does not arrive with a pre-labeled sentiment tag — that tag would itself need to be inferred from the ticket note.

Two viable options:

**Option A1 — Sentiment Tag in Branch 1 (Structured), excluded from Branch 2 (NLP)**
The tag is treated as a structured categorical feature alongside contract type, tenure, etc. Branch 2 uses only the raw PCA embeddings from `ticket_note`. This is the most defensible design: Branch 1 tests the maximum value of structured + qualitative signals, Branch 2 tests pure NLP signal, and the fusion measures whether combining them adds anything beyond what sentiment alone achieves.

**Option A2 — Sentiment Tag excluded from both branches, used only as a diagnostic feature**
Removes the tag entirely from training. Branch 1 uses only the 19 raw structured features. This is the strictest leakage-free baseline. The tag's correlation with the target makes it act as a near-perfect predictor, which would artificially inflate Branch 1 performance and obscure whether the NLP embeddings add genuine value.

**Recommendation: Option A2.** The goal of the Late Fusion experiment is to isolate the contribution of `ticket_note` embeddings. Including a near-deterministic proxy of the target in Branch 1 would make the baseline artificially strong and undermine the ROI narrative. The sentiment tag is better used as a diagnostic artifact — logged to MLflow as a standalone feature importance reference, not as a training signal.

---

### Decision 2 — SMOTE Placement

Class imbalance is 2.77:1 (Churn=No: 73.5%, Churn=Yes: 26.5%). Two options:

**Option B1 — SMOTE applied independently per branch**
Each branch's training data is oversampled separately after branch-specific feature extraction. Clean separation, branch-specific imbalance handling.

**Option B2 — SMOTE applied once on the full training set before branch split**
Single oversampling step on all features together before splitting into structured/NLP columns. Simpler, but SMOTE operates in a mixed feature space (structured + embeddings) which is geometrically less meaningful — synthetic samples generated across 19 structured + 20 NLP dimensions simultaneously.

**Recommendation: Option B1.** Apply SMOTE independently within each branch's training pipeline after feature extraction. Each branch operates in its own geometric space, so synthetic neighbor computation is more meaningful. It also keeps each branch's pipeline fully self-contained, which is required for the OOF cross-validation stacking protocol.
