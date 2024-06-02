import requests

# Send a GET request
print("Sending GET request...")
url = "http://127.0.0.1:8000/"
response = requests.get(url)
print("GET Request Status Code:", response.status_code)
print("Welcome Message:", response.text)

# Sample data for POST request
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Send a POST request
print("\nSending POST request...")
url = "http://127.0.0.1:8000/data/"
response_post = requests.post(url, json=data)
print("POST Request Status Code:", response_post.status_code)
print("Result:", response_post.text)