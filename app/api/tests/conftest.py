import pytest
from pymongo import AsyncMongoClient
import json
import os



@pytest.fixture(scope="module")
def set_env():
    
    os.environ["MONGODB_URL"] = "mongodb://localhost:27018/caas"

@pytest.fixture(scope="module")
async def test_client(set_env):
    client = AsyncMongoClient("mongodb://localhost:27018/caas")
    yield client
    await client.drop_database("caas")  # Clean up after all tests
    await client.close()

@pytest.fixture(scope="function")
async def test_db(test_client):
    db = test_client["caas"]
    result_collection = db["results"]
    
    with open("./app/api/tests/res/clustering_result.json", "r") as file:
        mock_result = json.load(file)
        await result_collection.insert_one(mock_result)

    yield db  # Provide the database to the test

    # Clean up after each test if needed
    await db["results"].delete_many({})
