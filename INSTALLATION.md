# Installation Guide

Complete setup instructions for the Vector Database Comparison project.

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **Docker**: 20.10 or higher
- **RAM**: 16 GB
- **Disk Space**: 40 GB free
  - 5 GB for Docker images
  - 35 GB for datasets and results

### Recommended Requirements
- **RAM**: 32 GB (for large datasets)
- **CPU**: 8+ cores
- **Disk**: SSD for better I/O performance
- **GPU**: NVIDIA GPU (optional, for GPU monitoring)

---

## Step 1: Install Prerequisites

### Windows

#### 1.1 Install Python
```powershell
# Download from python.org or use winget
winget install Python.Python.3.11

# Verify installation
python --version
```

#### 1.2 Install Docker Desktop
```powershell
# Download from docker.com or use winget
winget install Docker.DockerDesktop

# Start Docker Desktop
# Ensure WSL 2 backend is enabled in Docker settings
```

#### 1.3 Install Git (if needed)
```powershell
winget install Git.Git
```

### Linux

```bash
# Install Python
sudo apt update
sudo apt install python3.11 python3-pip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose

# Logout and login for group changes to take effect
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install Docker Desktop
brew install --cask docker
```

---

## Step 2: Clone/Download Project

```powershell
# If using Git
git clone <repository-url>
cd vector-db-comparison

# Or download and extract ZIP
# Then navigate to the directory
cd vector-db-comparison
```

---

## Step 3: Install Python Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pymilvus, weaviate; print('Dependencies OK')"
```

### Optional Dependencies

```powershell
# For GPU monitoring (NVIDIA GPUs only)
pip install pynvml

# For interactive dashboard (optional)
pip install streamlit
```

---

## Step 4: Start Vector Databases

### 4.1 Start Milvus

```powershell
# Navigate to docker directory
cd docker

# Start Milvus standalone
docker-compose -f docker-compose-milvus.yml up -d

# Wait for Milvus to be ready (~30 seconds)
# Check status
docker ps | findstr milvus

# Expected output: 3 containers running
# - milvus-standalone
# - milvus-etcd
# - milvus-minio
```

### 4.2 Start Weaviate

```powershell
# Start Weaviate
docker-compose -f docker-compose-weaviate.yml up -d

# Check status
docker ps | findstr weaviate

# Expected output: 1 container running
# - weaviate
```

### 4.3 Verify Connections

```powershell
# Return to project root
cd ..

# Test Milvus connection
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✓ Milvus connected')"

# Test Weaviate connection
python -c "import weaviate; client = weaviate.Client('http://localhost:8080'); print('✓ Weaviate connected')"
```

---

## Step 5: Run First Benchmark

### Quick Test (Synthetic Data)

```powershell
# Run basic benchmark with 100K synthetic vectors
python main.py
```

Expected output:
```
============================================================
MILVUS vs WEAVIATE BENCHMARK
============================================================

[1/6] Generating data...
[2/6] Loading data into Milvus...
[3/6] Loading data into Weaviate...
[4/6] Generating queries...
[5/6] Running benchmarks...
[6/6] Generating comparison report...

BENCHMARK COMPLETE
```

Results will be saved in `results/` directory.

---

## Step 6: Download Real Dataset (Optional)

```powershell
# List available datasets
python -m utils.dataset_downloader --list

# Download SIFT1M (~500 MB, recommended first dataset)
python -m utils.dataset_downloader --dataset sift1m

# This will download to data/datasets/sift1m/
# Download time: 5-30 minutes depending on connection
```

---

## Step 7: Run Real Data Benchmark (Optional)

```powershell
# Test with small subset first
python run_real_data_benchmark.py --dataset sift1m --subset 100000

# Run full SIFT1M benchmark
python run_real_data_benchmark.py --dataset sift1m
```

---

## Troubleshooting

### Issue: Docker containers won't start

**Solution**:
```powershell
# Check Docker is running
docker ps

# If error, start Docker Desktop
# Then retry

# Check for port conflicts
netstat -ano | findstr "19530"  # Milvus
netstat -ano | findstr "8080"   # Weaviate

# If ports are in use, stop conflicting services or modify docker-compose files
```

### Issue: Python dependencies fail to install

**Solution**:
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies one by one to identify issue
pip install numpy pandas
pip install pymilvus
pip install weaviate-client

# On Windows, if h5py fails:
pip install --only-binary h5py h5py
```

### Issue: "Permission denied" errors (Linux/macOS)

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login again

# Or run with sudo (not recommended)
sudo docker-compose up -d
```

### Issue: Out of memory during benchmark

**Solution**:
```powershell
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
# Set to at least 8 GB

# Or use smaller dataset
python run_real_data_benchmark.py --dataset sift1m --subset 50000
```

### Issue: Slow dataset download

**Solution**:
```powershell
# FTP downloads can be slow
# Alternative: Download manually from http://corpus-texmex.irisa.fr/
# Extract to data/datasets/sift1m/

# Or try different dataset
python -m utils.dataset_downloader --dataset glove-100  # HTTP download
```

### Issue: Milvus connection timeout

**Solution**:
```powershell
# Wait longer for Milvus to start (can take 1-2 minutes)
docker logs milvus-standalone

# Look for "Milvus Proxy successfully started"

# Restart if needed
docker-compose -f docker/docker-compose-milvus.yml restart
```

### Issue: Weaviate connection refused

**Solution**:
```powershell
# Check Weaviate logs
docker logs weaviate

# Restart if needed
docker-compose -f docker/docker-compose-weaviate.yml restart

# Verify port 8080 is accessible
curl http://localhost:8080/v1/.well-known/ready
```

---

## Verification Checklist

After installation, verify everything works:

- [ ] Python 3.8+ installed
- [ ] Docker Desktop running
- [ ] Python dependencies installed
- [ ] Milvus containers running (3 containers)
- [ ] Weaviate container running
- [ ] Milvus connection test passes
- [ ] Weaviate connection test passes
- [ ] Basic benchmark runs successfully
- [ ] Results appear in `results/` directory

---

## Uninstallation

### Stop and Remove Databases

```powershell
# Stop containers
docker-compose -f docker/docker-compose-milvus.yml down
docker-compose -f docker/docker-compose-weaviate.yml down

# Remove volumes (deletes all data)
docker-compose -f docker/docker-compose-milvus.yml down -v
docker-compose -f docker/docker-compose-weaviate.yml down -v

# Remove Docker images (optional)
docker rmi milvusdb/milvus:v2.3.0
docker rmi semitechnologies/weaviate:latest
```

### Remove Python Environment

```powershell
# Deactivate virtual environment
deactivate

# Remove virtual environment directory
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

---

## Next Steps

After successful installation:

1. **Read the README**: `README.md` for project overview
2. **Try Quick Reference**: `QUICK_REFERENCE.py` for common commands
3. **Run Advanced Analysis**: `python run_advanced_analysis.py`
4. **Read Advanced Usage**: `ADVANCED_USAGE.md` for detailed features

---

## Getting Help

If you encounter issues not covered here:

1. Check Docker logs: `docker logs <container-name>`
2. Check Python errors for missing dependencies
3. Verify system requirements are met
4. Try with smaller datasets first
5. Open an issue on GitHub (if applicable)

---

## Performance Tips

### For Faster Benchmarks
- Use SSD for Docker volumes
- Allocate more RAM to Docker (16+ GB)
- Close other applications during benchmarks
- Use smaller subsets for testing

### For Better Results
- Let databases warm up (run queries before benchmarking)
- Run multiple iterations and average results
- Monitor system resources during benchmarks
- Ensure no other heavy processes are running

---

**Installation Complete!** You're ready to benchmark vector databases.
