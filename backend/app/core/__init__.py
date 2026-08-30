# Cross-cutting application concerns that every layer may depend on:
# configuration (config.py) and the mapping from domain exceptions to HTTP
# responses (exceptions.py). Named `core` rather than `config` (its
# predecessor) because it now holds more than settings.
