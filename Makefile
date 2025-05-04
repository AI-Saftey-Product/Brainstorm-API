


.PHONY: generate_migrations
generate_migrations:
	alembic revision --autogenerate

.PHONY: apply_migrations
apply_migrations:
	alembic upgrade head

.PHONY: migrate
migrate:
	make generate_migrations
	make apply_migrations

.PHONY: run
run:
	uvicorn brainstorm.core.main:app --reload

.PHONY: cloud-sql-proxy
cloud-sql-proxy:
	./cloud-sql-proxy --address 0.0.0.0 -p 5633 hirundo-trial:us-central1:main-db


.PHONY: deploy
deploy:
	poetry export -f requirements.txt --output requirements.txt
	gcloud app deploy --quiet