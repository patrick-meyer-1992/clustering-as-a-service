import pytest
from httpx import ASGITransport, AsyncClient

from ..main import app


@pytest.mark.anyio
async def test_result_available(test_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/cluster/d9f0afa8-1d24-4fb7-950d-99507c84a010")
    assert response.status_code == 200
