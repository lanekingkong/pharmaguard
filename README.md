<div align="center">

# 💊 PharmaGuard

### Intelligent Medication Safety & DDI Prediction System

*AI-Powered Clinical Decision Support for Safer Polypharmacy*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

</div>

---

## The Clinical Problem

Globally, **adverse drug events (ADEs)** affect over **2 million patients annually** in the US alone, causing ~770,000 injuries. **Polypharmacy** (taking 5+ medications) affects 30-40% of elderly patients. The risk of drug-drug interactions (DDIs) grows exponentially with each additional medication, yet existing clinical decision support tools suffer from ~90% alert override rates due to false positives.

**PharmaGuard** is an open-source, AI-powered system that solves this by combining Graph Neural Networks, explainable AI, and multi-agent clinical reasoning.

## 5W1H Architecture

| Dimension | Description |
|-----------|-------------|
| **WHAT** | AI system detecting DDIs, assessing polypharmacy risks, providing explainable recommendations |
| **WHY** | ADEs cause ~770K injuries/year; existing CDS tools have 90% false positive rates; clinicians need trustworthy AI |
| **WHO** | Hospital pharmacists, clinical pharmacologists, physicians, geriatric specialists, clinical researchers |
| **WHERE** | Hospital EHR systems, pharmacy dispensing, clinical dashboards, mobile clinical apps |
| **WHEN** | Real-time at prescribing, medication reconciliation, routine polypharmacy review |
| **HOW** | GAT + Multi-Agent AI + Drug Knowledge Graph + SHAP Explainability + FastAPI/Streamlit deployment |

## Key Features

- **GNN DDI Prediction**: Multi-modal fusion (molecular fingerprints + protein targets + pathways + side effects + clinical context) using Graph Attention Networks with >85% accuracy
- **Polypharmacy Risk Scoring**: Multi-dimensional assessment incorporating drug count, patient age, renal/hepatic function, genetic variants, and comorbidities
- **Drug Knowledge Graph**: Neo4j-based graph with 6 node types and 10+ relationship types from DrugBank, PubMed, and FDA data
- **SHAP Explainability**: Feature-level importance, counterfactual explanations, and clinical reports for every prediction
- **Multi-Agent Clinical System**: Coordinator, DDI Analyst, Risk Assessor, Clinical Advisor, and Patient Educator agents
- **Full-Stack Deployment**: FastAPI REST API, Streamlit dashboard, Docker Compose, CLI tool

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/pharmaguard/pharmaguard.git
cd pharmaguard
docker compose up -d
# API: http://localhost:8000/api/docs | Dashboard: http://localhost:8501 | Neo4j: http://localhost:7474
```

### Local
```bash
pip install -e ".[dashboard]"
pharmaguard serve --port 8000 &
pharmaguard dashboard
```

### Python API
```python
from pharmaguard.saferx_agent.clinical.coordinator import SafeRxAgent
agent = SafeRxAgent()
result = agent.analyze_prescription_sync(
    patient_info={"age": 72, "egfr": 55.0, "comorbidities": ["hypertension"]},
    medications=[{"name": "warfarin", "dose": 5.0, "unit": "mg"}, {"name": "aspirin", "dose": 100.0, "unit": "mg"}]
)
```

### CLI
```bash
pharmaguard predict --drug-a warfarin --drug-b aspirin
pharmaguard analyze -m warfarin -m aspirin -m metformin --age 72 --egfr 55
pharmaguard query metformin
```

## Architecture

```
Web Dashboard (Streamlit) / CLI (Click) / REST API (FastAPI)
        │
┌───────▼────────── SafeRx Multi-Agent Clinical System ──────────┐
│ Coordinator → DDI Analyst → Risk Assessor → Clinical Advisor   │
└───────┬────────────────────────────┬──────────────────────────┘
        │                            │
   ┌────▼─────┐              ┌──────▼──────────┐
   │ DDI Model │              │  SHAP Explainer │
   │  (GAT)    │              │  (XAI)          │
   └────┬─────┘              └─────────────────┘
        │
   ┌────▼──────────────────────┐
   │  Drug Knowledge Graph     │
   │  (Neo4j: Drug↔Protein↔    │
   │   Disease↔Pathway↔Gene)   │
   └───────────────────────────┘
```

## Technology Stack
| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, PyG, GAT |
| Knowledge Graph | Neo4j, py2neo |
| XAI | SHAP |
| Cheminformatics | RDKit |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Multi-Agent | Async Python |
| Deployment | Docker, Compose |

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict/ddi` | Single DDI prediction |
| POST | `/api/v1/predict/ddi/batch` | Batch DDI prediction |
| POST | `/api/v1/analyze/prescription` | Full prescription analysis |
| POST | `/api/v1/query/drug` | Drug info query |
| GET | `/api/v1/drugs/search` | Drug search |

## Project Structure
```
pharmaguard/
├── src/pharmaguard/
│   ├── ddi_detector/models/gat_model.py    # GAT-based DDI prediction
│   ├── risk_assessor/scoring/polypharmacy_scorer.py  # Risk scoring
│   ├── medkg/kg_builder/knowledge_graph.py  # Neo4j knowledge graph
│   ├── xai_explainer/shap_explainer.py      # SHAP explainability
│   ├── saferx_agent/clinical/coordinator.py # Multi-agent system
│   ├── api/api_server.py                    # FastAPI server
│   ├── dashboard/app.py                     # Streamlit dashboard
│   ├── cli.py                               # CLI tool
│   └── config.py                            # Configuration
├── tests/
│   └── test_core.py                         # Unit/API tests
├── docker-compose.yml                       # Full-stack deployment
├── Dockerfile                               # Container build
├── pyproject.toml                           # Project metadata
├── setup.py                                 # Setup script
├── .env.example                             # Environment template
├── .gitignore
├── LICENSE                                  # MIT
└── README.md                                # This file
```

## Configuration
Copy `.env.example` to `.env` and configure Neo4j connection, API ports, model settings.

## Contributing
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

## Version Update Guide
1. Update `__version__` in `src/pharmaguard/__init__.py`
2. Update `version` in `pyproject.toml`
3. Run tests: `pytest tests/ -v`
4. Rebuild Docker: `docker compose build`
5. Tag: `git tag v1.x.x && git push --tags`

## Citation
```bibtex
@software{pharmaguard2025,
  author = {PharmaGuard Team},
  title = {PharmaGuard: Intelligent Medication Safety & DDI Prediction System},
  year = {2025},
  url = {https://github.com/pharmaguard/pharmaguard},
  version = {1.0.0}
}
```

## License
MIT License - see [LICENSE](LICENSE)

## Acknowledgments
DrugBank, FDA FAERS, PubChem, SIDER, PyTorch Geometric, SHAP, RDKit, Neo4j, FastAPI, Streamlit

---

<div align="center">
<b>PharmaGuard</b> — Making polypharmacy safer through AI.<br>
<i>"First, do no harm" — now empowered by artificial intelligence.</i>
</div>