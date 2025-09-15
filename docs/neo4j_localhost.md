# Neo4j localhost for Graphiti usage

```yaml
services:
  neo4j:
    image: neo4j:5.26.8 # LTS version
    volumes:
        - ./neo4j/logs:/logs
        - ./neo4j/config:/config
        - ./neo4j/data:/data
        - ./neo4j/plugins:/plugins
    ports:
      - 7473:7473
      - 7474:7474
      - 7687:7687 # web UI
    environment:
        - NEO4J_AUTH=neo4j/12345678 # user/password
        - NEO4J_PLUGINS=["graph-data-science"]
    # restart: always
```
