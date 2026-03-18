```mermaid
flowchart TD
	node1["data_ingestion"]
	node2["enrich_data"]
	node3["feature_engineering"]
	node4["validate_enriched"]
	node5["validate_raw"]
	node1-->node2
	node1-->node5
	node2-->node3
	node2-->node4
```