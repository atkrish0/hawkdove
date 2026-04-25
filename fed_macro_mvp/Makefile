up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f

telegraf-logs:
	docker compose logs -f telegraf

questdb-logs:
	docker compose logs -f questdb

grafana-logs:
	docker compose logs -f grafana

count:
	curl -s "http://localhost:9000/exec?query=select%20count()%20from%20rag_events"

endpoints:
	@echo "Grafana: http://localhost:3000"
	@echo "QuestDB UI: http://localhost:9000"
	@echo "QuestDB ILP: localhost:9009"
	@echo "QuestDB Postgres: localhost:8812"
