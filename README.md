
## Langgraph Agent
Navigate to the agent directory:
```bash
cd ~/git_repos/ag_ui_compatible_server/langgraph_agents
```
To start the Langgraph Agent, execute:
```bash
uvicorn server:app --reload --port 2024
```
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres

## Langgraph Agent Development
Ensure you are in the correct directory:
```bash
cd ~/git_repos/ag_ui_compatible_server/langgraph_agents
```
For development mode, use the following command:
```bash
langgraph dev --host 0.0.0.0 --allow-blocking
```