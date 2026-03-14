import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        # If no api_key configured, allow all requests
        if self._api_key is None:
            return await call_next(request)

        # Validate Authorization: Bearer <key>
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {self._api_key}":
            return await call_next(request)

        # Return 401 with JSON error
        return Response(
            content=json.dumps({"success": False, "message": "Invalid API key"}),
            status_code=401,
            media_type="application/json",
        )
