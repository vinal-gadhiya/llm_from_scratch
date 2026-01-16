def test_health_endpoint_exist(client):
    response = client.get("/health")
    assert response.status_code in [200, 503]

def test_health_response_structure(client):
    response = client.get("/health")
    data = response.json()

    if response.status_code == 200:
        assert "status" in data
        assert "model_loaded" in data
        assert data["model_loaded"] is True