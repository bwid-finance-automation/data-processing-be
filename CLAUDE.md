# Data Processing Backend

## What
N-Layer Monolith FastAPI backend for financial data processing automation.

**Stack**: Python 3.13, FastAPI, SQLAlchemy (async) + PostgreSQL, openpyxl, OpenAI/Anthropic/Gemini APIs

**Project structure**:
```
app/
  presentation/api/     → API routers (REST endpoints)
  application/          → Use cases, orchestration, business workflows
  domain/               → Business logic, models, domain services
  infrastructure/       → DB, external APIs, caching (Redis)
  shared/               → Exceptions, utils, prompts
  core/                 → App config, exception handlers
templates/              → Excel templates (cash report, etc.)
alembic/                → DB migrations
```

**Modules**:
- `finance/` — Bank statement parsing, cash report automation, contract OCR, utility billing, variance analysis
- `fpa/` — Excel comparison, GLA variance, NTM EBITDA

## How

**Run locally**: `python main.py` (port 8000)
**Run with Docker**: `./RUN_DOCKER.sh`
**DB migrations**: `alembic revision --autogenerate -m "message"` then `alembic upgrade head`
**API docs**: http://localhost:8000/docs

## Critical: Excel file manipulation
This project manipulates .xlsx files at the **ZIP/XML level** to preserve drawings, charts, and shapes. Never use `openpyxl.load_workbook() + wb.save()` for modifying files — it destroys embedded objects. Details in `memory/MEMORY.md`.
