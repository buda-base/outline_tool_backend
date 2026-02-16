class NotFoundError(Exception):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, resource_id: str) -> None:
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} '{resource_id}' not found")


class ConflictError(Exception):
    """Raised when an operation conflicts with the current state of a resource."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
