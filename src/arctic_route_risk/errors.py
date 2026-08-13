"""Work Package B failures with stable machine-readable prefixes."""


class RiskPipelineError(ValueError):
    """Base error for rejected input, build, or publication operations."""


class InputIdentityError(RiskPipelineError):
    """RunContext, DatasetBundle, PreparedWindow, or frame identity disagrees."""


class CoverageError(RiskPipelineError):
    """The requested formal window cannot be produced without hiding a gap."""


class GridCompatibilityError(RiskPipelineError):
    """A source grid cannot be transformed by the declared grid policy."""


class StaleGenerationError(RiskPipelineError):
    """A task attempted to publish outside the active simulation generation."""


class PublicationConflictError(RiskPipelineError):
    """An immutable ID or commit ID already points at different content."""
