"""Pydantic schemas for structured LLM output.

`copilot.py` defines the planner's structured plan and the Copilot's
final response. A node that wants structured output from the model
should define the schema here rather than inline, the same way
`llm/model.py` centralizes chat-model construction instead of letting
each node build its own client.
"""
