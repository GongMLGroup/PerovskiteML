# PerovskiteML
Machine learning for perovskite photovoltaics research. Supervised by [Dr. Jiawei Gong](https://scholar.google.com/citations?user=EHYyzbIAAAAJ).

---

## Installation
1. Install [uv](https://docs.astral.sh/uv/) (Python package manager):
   ```bat
   winget install --id=astral-sh.uv -e
   ```
2. Clone and install dependencies:
   ```bat
   git clone https://github.com/GongMLGroup/PerovskiteML
   cd PerovskiteML
   uv sync
   ```
   This creates a `.venv` virtual environment. VS Code will automatically detect it.

---

## Updating
After pulling changes:
```bat
git pull && uv sync
```

---

## Neptune.ai Setup (Recommended)
To enable experiment tracking, set your API token:

### Windows Instructions
**Temporary (per session):**
```bat
:: Command Prompt
set NEPTUNE_API_TOKEN=your_api_key_here
```
```powershell
# PowerShell
$env:NEPTUNE_API_TOKEN = "your_api_key_here"
```

**Permanent:**
```bat
:: Command Prompt (system-wide)
setx NEPTUNE_API_TOKEN "your_api_key_here"
```
```powershell
# PowerShell (user-level)
[Environment]::SetEnvironmentVariable("NEPTUNE_API_TOKEN", "your_api_key_here", "User")
```
Restart your terminal after setting permanently.

### macOS/Linux (Optional)
```bash
# Temporary
export NEPTUNE_API_TOKEN="your_api_key_here"
```
```bash
# Permanent
echo 'export NEPTUNE_API_TOKEN="your_api_key_here"' >> ~/.bashrc && source ~/.bashrc
```

---

## Verification
Check if token is set:
```bat
:: Command Prompt
echo %NEPTUNE_API_TOKEN%
```
```powershell
# PowerShell
echo $env:NEPTUNE_API_TOKEN
```

**Note:** Replace `your_api_key_here` with your actual token. Never commit API keys to version control.
