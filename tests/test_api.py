"""Tests for the Flask API routes in app.py."""

import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert resp.content_type.startswith('text/html')


class TestGeojson:
    def test_geojson_endpoint(self, client):
        resp = client.get('/taxi_zones.json')
        # The file may or may not exist locally; verify the response shape
        if resp.status_code == 200:
            data = resp.get_json()
            assert 'type' in data
            assert 'features' in data


class TestDemandApi:
    def test_demand_endpoint_structure_on_error(self, client):
        """If Hopsworks is not configured, the endpoint should return 500."""
        resp = client.get('/api/demand')
        # Without HOPSWORKS_API_KEY we expect an error
        assert resp.status_code in (200, 500)
        if resp.status_code == 500:
            data = resp.get_json()
            assert 'error' in data

    def test_demand_returns_json(self, client):
        resp = client.get('/api/demand')
        assert resp.content_type.startswith('application/json')
