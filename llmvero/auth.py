import hmac

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


def require_api_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Accept `Authorization: Bearer <key>` or `X-API-Key: <key>` when ACCESS_KEY is set."""
    expected = settings.access_key
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    supplied = (
        auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    )
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
