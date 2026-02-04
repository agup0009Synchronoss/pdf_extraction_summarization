# Ollama Service Management Commands

## Prerequisites
First, ensure kubectl is in your PATH and KUBECONFIG is set:
```powershell
# Add kubectl to PATH (if not already done)
$env:PATH += ";C:\Users\agup0009\AppData\Local\Microsoft\WinGet\Packages\Kubernetes.kubectl_Microsoft.Winget.Source_8wekyb3d8bbwe"

# Set KUBECONFIG
$env:KUBECONFIG = "C:\Users\agup0009\code\PDF\k8s-config"
```

## Service Status Commands

### Check Current Status
```powershell
# Check all pods in ml-ollama namespace
kubectl -n ml-ollama get pods

# Check deployment status
kubectl -n ml-ollama get deployments

# Check services
kubectl -n ml-ollama get services

# Check cronjobs
kubectl -n ml-ollama get cronjobs

# Check events (useful for troubleshooting)
kubectl -n ml-ollama get events --sort-by='.lastTimestamp'
```

### Detailed Status Information
```powershell
# Get detailed pod information
kubectl -n ml-ollama describe pods

# Get detailed deployment information
kubectl -n ml-ollama describe deployments

# Get logs from running pods
kubectl -n ml-ollama logs -l app=ollama-gpu-cronjobs
```

## Service Control Commands

### Scale Up (Launch) the Service
```powershell
# Scale up both deployments to 1 replica
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs --replicas=1
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs-auth --replicas=1

# Alternative: Scale up all deployments in the namespace
kubectl -n ml-ollama scale deployment --all --replicas=1
```

### Scale Down (Close) the Service
```powershell
# Scale down both deployments to 0 replicas
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs --replicas=0
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs-auth --replicas=0

# Alternative: Scale down all deployments in the namespace
kubectl -n ml-ollama scale deployment --all --replicas=0
```

### Restart the Service
```powershell
# Restart deployments (rollout restart)
kubectl -n ml-ollama rollout restart deployment ml-ollama-ollama-gpu-cronjobs
kubectl -n ml-ollama rollout restart deployment ml-ollama-ollama-gpu-cronjobs-auth

# Check rollout status
kubectl -n ml-ollama rollout status deployment ml-ollama-ollama-gpu-cronjobs
```

## Monitoring Commands

### Watch Service Status in Real-time
```powershell
# Watch pods status
kubectl -n ml-ollama get pods -w

# Watch deployments status
kubectl -n ml-ollama get deployments -w

# Watch events
kubectl -n ml-ollama get events -w
```

### Resource Usage
```powershell
# Check resource usage
kubectl -n ml-ollama top pods
kubectl -n ml-ollama top nodes
```

## Gradio App Launch Commands

### Launch with Public Access
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Launch Gradio app with public access and shareable link
python gradio_app.py --share --port 7860

# Alternative: Use default port (7861)
python gradio_app.py --share

# Alternative: Use custom port
python gradio_app.py --share --port 8080
```

### Launch with Custom Configuration
```powershell
# Launch with specific settings
python gradio_app.py --share --port 7860

# Launch with default port
python gradio_app.py --share

# Launch without share (local only)
python gradio_app.py --port 7860
```

### Launch with Environment Variables
```powershell
# Set environment variables and launch
$env:GRADIO_PORT = "7860"
$env:GRADIO_SHARE = "true"
python gradio_app.py
```

## Quick Status Check Script
Create a PowerShell script for quick status checking:

```powershell
# Save as check-ollama-status.ps1
Write-Host "=== Ollama Service Status ===" -ForegroundColor Green
Write-Host "Pods:" -ForegroundColor Yellow
kubectl -n ml-ollama get pods
Write-Host "`nDeployments:" -ForegroundColor Yellow
kubectl -n ml-ollama get deployments
Write-Host "`nServices:" -ForegroundColor Yellow
kubectl -n ml-ollama get services
```

## Troubleshooting Commands

### If Pods are Stuck
```powershell
# Check pod events
kubectl -n ml-ollama describe pod <pod-name>

# Check pod logs
kubectl -n ml-ollama logs <pod-name>

# Force delete stuck pod
kubectl -n ml-ollama delete pod <pod-name> --force --grace-period=0
```

### If Deployments are Stuck
```powershell
# Check deployment events
kubectl -n ml-ollama describe deployment <deployment-name>

# Rollback to previous version
kubectl -n ml-ollama rollout undo deployment <deployment-name>

# Check rollout history
kubectl -n ml-ollama rollout history deployment <deployment-name>
```

## Current Status Summary
- **Service**: Partially running (1/2 deployments active)
- **Auth Service**: ✅ Running (1/1 replicas)
- **Main Service**: ⏳ Pending (0/1 replicas - may be waiting for resources)
- **Last Action**: Scaled up manually
- **Next Auto-scale**: Monday at 10:59 AM (weekly schedule)
