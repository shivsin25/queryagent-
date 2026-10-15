class QueryToolError(Exception):
    """Base error for the application."""


class LLMNotConfiguredError(QueryToolError):
    """The selected LLM provider is missing credentials or its package."""


class BudgetExceededError(QueryToolError):
    """Today's LLM spend has reached the configured daily budget."""


class RetryLimitError(QueryToolError):
    """A question has been answered again as many times as allowed."""


class NotFoundError(QueryToolError):
    """A referenced request does not exist."""


class DocumentError(QueryToolError):
    """A document could not be read or indexed."""


class PipelineError(QueryToolError):
    """The agent pipeline failed to produce an answer."""
