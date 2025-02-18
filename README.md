# PerovskiteML
Using machine learning to understand perovskite photovoltaics. Under the supervision of [Dr. Jiawei Gong.](https://scholar.google.com/citations?user=EHYyzbIAAAAJ)
## Setup Environment File
Add a file named `.env` to the root of the project.
```
📦PerovskiteML
 ┣ 📂data
 ┣ 📂docs
 ┣ 📂figures
 ┣ 📂julia
 ┣ 📂models
 ┣ 📂notebooks
 ┣ 📂perovskiteml
 ┣ 📜.env ⭐
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜pyproject.toml
 ┗ 📜uv.lock
 ```
 Inside the `.env` file add important information.
 ```js
# Neptune Information
NEPTUNE_PROJECT = "workspace/project-name"
NEPTUNE_API_TOKEN = "YOUR_API_TOKEN"
 ```

## Installation
Once you clone the repository you can use [uv](https://docs.astral.sh/uv/), a python package and project manager to install the project.
```bash
winget install --id=astral-sh.uv  -e
```
More installation methods can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

Navigate to your clone of the PerovskiteML Project and install it with the following command:
```bash
uv sync
```
This should install the dependencies to a virtual environment named `.venv`. VSCode will automatically detect and activate this envornment for you. 

## Update your Virtual Environment
Whenever there are updates to the project you can use:
```bash
git pull
```
To download and update the project files. And:
```bash
uv sync
```
To update the dependencies and virtual environment.
