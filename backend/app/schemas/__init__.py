# Pydantic models that define the shape of request/response bodies —
# the API's public contract. Kept separate from app/models/ (reserved for
# future database/ORM entities): a request schema and a database row often
# diverge (e.g. a schema hides internal fields, or accepts a field the
# database derives), so treating them as the same type from the start
# would make that divergence harder to introduce later.
