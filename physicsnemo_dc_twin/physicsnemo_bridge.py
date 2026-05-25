def physicsnemo_status():
    """Return import status without making PhysicsNeMo mandatory for the pilot."""
    try:
        import physicsnemo  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "message": f"PhysicsNeMo is not importable yet: {exc}",
        }

    return {
        "available": True,
        "version": getattr(physicsnemo, "__version__", "unknown"),
        "message": "PhysicsNeMo is importable. Native PhysicsNeMo backend can be used.",
    }
