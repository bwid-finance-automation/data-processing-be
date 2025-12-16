# Deploy to Render

## Quick Deploy (Blueprint)

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New** → **Blueprint**
4. Connect your GitHub repo
5. Render will auto-detect `render.yaml` and create:
   - PostgreSQL database
   - Web service

## Manual Deploy

### Step 1: Create PostgreSQL Database

1. Go to Render Dashboard → **New** → **PostgreSQL**
2. Configure:
   - Name: `data-processing-db`
   - Database: `data_processing`
   - User: `postgres`
   - Region: Singapore (or nearest)
   - Plan: Free (for testing) or Starter (for production)
3. Click **Create Database**
4. Copy the **Internal Database URL** (starts with `postgres://`)

### Step 2: Create Web Service

1. Go to Render Dashboard → **New** → **Web Service**
2. Connect your GitHub repository
3. Configure:
   - Name: `data-processing-api`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free or Starter

### Step 3: Set Environment Variables

In Web Service → **Environment**:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | (paste Internal Database URL from Step 1) |
| `DATABASE__SSL_MODE` | `require` |
| `DATABASE__POOL_SIZE` | `5` |
| `OPENAI_API_KEY` | (your OpenAI key) |
| `GEMINI_API_KEY` | (your Gemini key) |
| `VARIANCE_APP__DEBUG` | `false` |
| `VARIANCE_APP__LOG_LEVEL` | `INFO` |

### Step 4: Run Migrations

After first deploy, run migrations via Render Shell:

```bash
# In Render Dashboard → Web Service → Shell
alembic upgrade head
```

Or add to `render.yaml` build command:
```yaml
buildCommand: pip install -r requirements.txt && alembic upgrade head
```

## Environment Configuration

### Local Development
```env
DATABASE__HOST=localhost
DATABASE__PORT=5432
DATABASE__NAME=data_processing
DATABASE__USER=postgres
DATABASE__PASSWORD=your_password
DATABASE__SSL_MODE=prefer
```

### Render Production
```env
DATABASE_URL=postgres://user:pass@host:port/database
DATABASE__SSL_MODE=require
```

## Troubleshooting

### Connection Issues
- Ensure `DATABASE__SSL_MODE=require` is set
- Check if database is in same region as web service
- Use **Internal Database URL** (not External)

### Migration Errors
```bash
# Reset migrations if needed
alembic downgrade base
alembic upgrade head
```

### Performance
- Free tier: 256MB RAM, shared CPU
- Starter tier: 512MB RAM, 0.5 CPU
- Consider upgrading for production workloads

## Costs (as of 2024)

| Service | Free | Starter | Standard |
|---------|------|---------|----------|
| Web Service | $0 | $7/mo | $25/mo |
| PostgreSQL | $0 (90 days) | $7/mo | $20/mo |

Free tier limitations:
- Spins down after 15 min inactivity
- 750 hours/month
- PostgreSQL expires after 90 days
