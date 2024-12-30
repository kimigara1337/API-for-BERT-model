import pytest
from main import app, classifier
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_classifier():
    classifier.check_on_bad_request = MagicMock()
    yield classifier

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the FastAPI app! Use the /check_prompt endpoint."
    }

def test_check_prompt(mock_classifier):
    mock_classifier.check_on_bad_request.return_value = "safe"
    
    response = client.get("/check_prompt/", params={"prompt_input": "Hello, this is a test"})
    
    assert response.status_code == 200
    assert response.json() == {
        "result": "safe",
        "success": True,
    }
    mock_classifier.check_on_bad_request.assert_called_once_with("Hello, this is a test")