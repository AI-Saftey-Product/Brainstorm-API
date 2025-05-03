


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