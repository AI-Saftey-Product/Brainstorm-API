


.PHONY: generate_migrations
generate_migrations:
	alembic revision --autogenerate

.PHONY: apply_migrations
apply_migrations:
	alembic upgrade head