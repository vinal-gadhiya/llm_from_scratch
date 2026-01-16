import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from serving.main import app
    return TestClient(app)

@pytest.fixture
def sample_request():
    return {"user_input": "This is a test input"}