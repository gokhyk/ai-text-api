from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_empty_text_returns_400():
    r = client.post("/v1/summarize", json={"text": "   "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Text cannot be empty"

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "health ok"
