# PerovskiteML
Using machine learning to understand perovskite photovoltaics. Under the supervision of [Dr. Jiawei Gong.](https://scholar.google.com/citations?user=EHYyzbIAAAAJ)
## Setup Environment File
Add a file named `.env` to the root of the project.
```
📦PerovskiteML
 ┣ 📂data
 ┣ 📂figures
 ┣ 📂notebooks
 ┣ 📂src
 ┣ 📜.env ⭐
 ┣ 📜.gitignore
 ┣ 📜Project.toml
 ┣ 📜pyrequirements.txt
 ┗ 📜README.md
 ```
 Inside the `.env` file add important information.
 ```
# Neptune Information
NEPTUNE_PROJECT = "workspace/project-name"
NEPTUNE_API_TOKEN = "YOUR_API_TOKEN"
 ```

## Update your Virtual Environment
To make sure your packages are up to date make sure to update your virtual environment when syncing with new changes.
```bash
./.venv/Scripts/activate
pip install -r requirements.txt
```
