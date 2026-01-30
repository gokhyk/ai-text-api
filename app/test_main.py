from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_empty_text_returns_422():
    r = client.post("/summarize", json={"text": "   "})
    assert r.status_code == 422

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "health ok"
