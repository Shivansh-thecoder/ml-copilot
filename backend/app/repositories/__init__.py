# Storage abstractions. Each resource gets an abstract base class (the
# contract routers/services depend on) plus one concrete implementation —
# in-memory for now. Introducing this seam before a database exists means
# swapping in a SQLAlchemy-backed implementation later is a one-file change
# in app/api/dependencies.py, not a rewrite of the service or router layers.
