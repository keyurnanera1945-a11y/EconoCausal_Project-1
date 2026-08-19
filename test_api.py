"""
EconoCausal - Week 4
REST API tests.
"""

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


print("=" * 70)
print("ECONOCAUSAL - WEEK 4 API TEST")
print("=" * 70)


# --------------------------------------------------
# Test 1
# --------------------------------------------------

print("\n1. Testing root endpoint...")

response = client.get("/")

print(
    "Status:",
    response.status_code
)

print(
    response.json()
)

assert response.status_code == 200


# --------------------------------------------------
# Test 2
# --------------------------------------------------

print("\n2. Testing health endpoint...")

response = client.get("/health")

print(
    "Status:",
    response.status_code
)

print(
    response.json()
)

assert response.status_code == 200

assert (
    response.json()["status"]
    == "healthy"
)


# --------------------------------------------------
# Test 3
# --------------------------------------------------

print("\n3. Testing dataset endpoint...")

response = client.get("/dataset")

print(
    "Status:",
    response.status_code
)

if response.status_code == 200:

    print(
        response.json()
    )


# --------------------------------------------------
# Test 4
# --------------------------------------------------

print("\n4. Testing prediction endpoint...")

customer = {

    "recency": 5,

    "history": 250,

    "mens": 1,

    "womens": 1,

    "newbie": 0,
}


response = client.post(
    "/predict",
    json=customer
)


print(
    "Status:",
    response.status_code
)

print(
    response.json()
)

assert response.status_code == 200


# --------------------------------------------------
# Test 5
# --------------------------------------------------

print("\n5. Testing invalid input...")

invalid_customer = {

    "recency": -10,

    "history": 250,

    "mens": 1,

    "womens": 1,

    "newbie": 0,
}


response = client.post(
    "/predict",
    json=invalid_customer
)


print(
    "Status:",
    response.status_code
)

assert response.status_code == 422


print("\nAll API tests passed!")

print("=" * 70)

print(
    "WEEK 4 API TEST COMPLETED SUCCESSFULLY"
)

print("=" * 70)