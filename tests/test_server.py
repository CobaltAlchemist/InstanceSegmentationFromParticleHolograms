import pytest
import time
from flask import Flask
from common import config
from fakeholo.server import genserver
import logging

logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def client():
    with genserver.test_client() as client:
        yield client
    client.post('/close')

@pytest.fixture(scope="module")
def configured_client(config):
    with genserver.test_client() as client:
        client.post('/set_config', json=config)
        while client.get('/get_sample').status_code == 400:
            time.sleep(1)
        yield client
    client.post('/close')

def test_set_config(client, config):
    response = client.post('/set_config', json=config)
    assert response.status_code == 200
    assert response.json == {"message": "Configuration updated and buffer reset."}

    response = client.post('/set_config', json=config)
    assert response.status_code == 200
    assert response.json == {"message": "Configuration unchanged."}

def test_gives_error_when_filling(client, config):
    client.post('/set_config', json=config)
    response = client.get('/get_sample')
    time.sleep(1)
    assert response.status_code == 400
    assert response.json == {"error": "Not enough samples in buffer, try again later"}


def test_get_sample(configured_client):
    response = configured_client.get('/get_sample')
    assert response.status_code == 200
    assert response.content_type == "application/octet-stream"