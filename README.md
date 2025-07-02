# Multi-Agent RAG System (Azure ML + SharePoint)

## Overview
This project implements a multi-agent Retrieval-Augmented Generation (RAG) system that integrates with Azure ML (Fabric datasets) and SharePoint for enterprise knowledge retrieval, using Streamlit for the user interface.

## Features
- **Planner Agent**: Selects the best agents for each query.
- **FabricDataRetrievalAgent**: Retrieves structured data from Azure ML datasets.
- **SharePointDataRetrievalAgent**: Retrieves documents from SharePoint.
- **Verifier Agent**: Validates and refines responses.
- **Summary Agent**: Summarizes results.

## Folder Structure
```
gbb-ai-agenticrag-main/
├── data/
├── src/
│   ├── aoai/
│   ├── azureaiagents/
│   └── azureAIsearch/
├── usecases/
│   └── agenticrag/
│       ├── app.py
│       ├── prompts.py
│       ├── settings.py
│       ├── state.py
│       ├── tools.py
│       └── aoaiAgents/
│           └── agent_store/
│               ├── planner_agent.yaml
│               ├── summary_agent.yaml
│               └── verifier_agent.yaml
├── utils/
├── requirements.txt
├── agents_sample.csv
└── README.md
```

## Setup
1. **Clone the repo**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   pip install Office365-REST-Python-Client
   ```
3. **Configure environment variables**
   - Azure ML credentials (see `.env.example`)
   - SharePoint credentials (site URL, username, password)
4. **Edit `usecases/agenticrag/tools.py`**
   - Set your dataset name for Fabric
   - Set your SharePoint site URL and credentials
5. **Run the app**
   ```sh
   streamlit run usecases/agenticrag/app.py
   ```

## Example Queries
- "Show me the sales figures from Fabric datasets for Q4 2023."
- "Find the latest compliance report for Product X in SharePoint."
- "Compare ProductA and ProductB accuracy using all available data."

## Notes
- For production, use app registration and client credentials for SharePoint.
- Extend `tools.py` to add more agents or connect to other APIs.

---
For more details, see the code comments and inline documentation.
