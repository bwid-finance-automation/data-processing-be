# Quick Command Reference - BWID Backend

## 🚀 **Easiest Way: Use the Script**

```bash
cd "/Users/andyl/Desktop/BWID/BWID Automation/data-processing-be"

# Run the backend (builds if needed)
./RUN_DOCKER.sh

# Rebuild and run
./RUN_DOCKER.sh rebuild

# Stop the backend
./RUN_DOCKER.sh stop

# View logs
./RUN_DOCKER.sh logs
```

---

## 📋 **Manual Commands (If You Prefer)**

### **1. Build the Image**

```bash
cd "/Users/andyl/Desktop/BWID/BWID Automation/data-processing-be"

# Build (takes 3-5 minutes first time)
docker build -t bwid-backend:latest .
```

### **2. Run the Container**

```bash
# Start backend
docker run -d \
  --name bwid-backend \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/data:/app/data" \
  bwid-backend:latest
```

### **3. Check Status**

```bash
# Check if running
docker ps | grep bwid-backend

# View logs (last 50 lines)
docker logs --tail 50 bwid-backend

# Follow logs in real-time
docker logs -f bwid-backend

# Check health
curl http://localhost:8000/
```

### **4. Stop/Restart**

```bash
# Stop
docker stop bwid-backend

# Start again (after stop)
docker start bwid-backend

# Restart
docker restart bwid-backend

# Stop and remove
docker stop bwid-backend && docker rm bwid-backend
```

### **5. Access Container Shell**

```bash
# Enter container
docker exec -it bwid-backend bash

# Once inside, you can:
tesseract --version          # Check Tesseract
printenv | grep OPENAI       # Check API key
ls -la /app                  # View files
exit                         # Exit shell
```

---

## 🧪 **Test the API**

### **Quick Health Check**

```bash
# Test root endpoint
curl http://localhost:8000/

# Test Contract OCR endpoint
curl http://localhost:8000/api/finance/contract-ocr/

# Test health
curl http://localhost:8000/api/finance/health
```

### **View Documentation**

```bash
# Open Swagger UI (interactive)
open http://localhost:8000/docs

# Open ReDoc (pretty docs)
open http://localhost:8000/redoc
```

### **Process a Contract**

```bash
# Process single contract
curl -X POST "http://localhost:8000/api/finance/contract-ocr/process-contract" \
  -F "file=@/path/to/contract.pdf"

# Export to Excel
curl -X POST "http://localhost:8000/api/finance/contract-ocr/export-to-excel" \
  -F "files=@contract1.pdf" \
  -F "files=@contract2.pdf" \
  --output contracts.xlsx
```

---

## 🔧 **Troubleshooting Commands**

### **Container won't start?**

```bash
# Check logs for errors
docker logs bwid-backend

# Check if port is already in use
lsof -i :8000

# Remove old containers
docker rm -f bwid-backend
```

### **API not responding?**

```bash
# Check container is running
docker ps | grep bwid

# Check health status
docker inspect bwid-backend --format='{{.State.Health.Status}}'

# Enter container and test
docker exec -it bwid-backend bash
curl localhost:8000/
```

### **Need to rebuild?**

```bash
# Remove old image and rebuild
docker stop bwid-backend
docker rm bwid-backend
docker rmi bwid-backend:latest
docker build -t bwid-backend:latest .
```

---

## 📊 **Monitoring**

```bash
# Container stats (CPU, memory)
docker stats bwid-backend

# Disk usage
docker exec bwid-backend du -sh /app/outputs/*

# Process list
docker top bwid-backend
```

---

## 🧹 **Cleanup**

```bash
# Stop and remove container
docker stop bwid-backend
docker rm bwid-backend

# Remove image
docker rmi bwid-backend:latest

# Clean up all unused Docker resources
docker system prune -a
```

---

## 📝 **Common Workflows**

### **Daily Use:**

```bash
# Start
./RUN_DOCKER.sh

# Check it's working
curl http://localhost:8000/api/finance/contract-ocr/

# View logs if needed
./RUN_DOCKER.sh logs

# Stop when done
./RUN_DOCKER.sh stop
```

### **After Code Changes:**

```bash
# Rebuild and restart
./RUN_DOCKER.sh rebuild
```

### **Check Logs for Errors:**

```bash
# View all logs
docker logs bwid-backend

# Filter for errors only
docker logs bwid-backend 2>&1 | grep -i error

# Filter for contract processing
docker logs bwid-backend 2>&1 | grep -i contract
```

---

## 🎯 **Quick Reference**

| Task | Command |
|------|---------|
| Start backend | `./RUN_DOCKER.sh` |
| Stop backend | `./RUN_DOCKER.sh stop` |
| View logs | `./RUN_DOCKER.sh logs` |
| Rebuild | `./RUN_DOCKER.sh rebuild` |
| Check status | `docker ps \| grep bwid` |
| Test API | `curl http://localhost:8000/` |
| View docs | `open http://localhost:8000/docs` |
| Container shell | `docker exec -it bwid-backend bash` |

---

**That's it!** Use `./RUN_DOCKER.sh` for the easiest experience, or use the manual commands if you prefer more control.
