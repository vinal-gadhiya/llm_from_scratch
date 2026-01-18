def test_serving_endpoint_method_not_allowed(client):
    response = client.get("/chat/model_inference")
    assert response.status_code == 405

def test_predict_endpoint_structure(client, sample_request):
    response = client.post("/chat/model_inference", json=sample_request)
    assert response.status_code in [200, 422, 503]

def test_predict_invalid_input(client):
    response = client.post("/chat/model_inference", json={"user_input": "This is random text", "user_input2": 5000})
    assert response.status_code in [422, 503]

def test_empty_input(client):
    response = client.post("/chat/model_inference", json={"user_input": ""})
    assert response.status_code in [200, 422, 503]

def test_model_response_structure(client, sample_request):
    response = client.post("/chat/model_inference", json=sample_request)
    if response.status_code == 200:
        data = response.json()
        assert "model_output" in data
        assert isinstance(data["model_output"], str)