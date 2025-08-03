import pandas as pd
import requests
import json

# -------------------- Configuration --------------------
API_KEY = "ApiKey-699eac6f-b1c9-4407-ac7b-1e08242972bd"

# You can use either public or private endpoint
DEPLOYMENT_URL_PUBLIC = "https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/6a329373-216e-4b77-bb42-a7eb799a997a/predictions?version=2021-05-01"
DEPLOYMENT_URL_PRIVATE = "https://private.au-syd.ml.cloud.ibm.com/ml/v4/deployments/6a329373-216e-4b77-bb42-a7eb799a997a/predictions?version=2021-05-01"

USE_PRIVATE_URL = False  # Change to True if using the private endpoint

DEPLOYMENT_URL = DEPLOYMENT_URL_PRIVATE if USE_PRIVATE_URL else DEPLOYMENT_URL_PUBLIC
CSV_FILE_PATH = "DistrictswiseCR_AEdataf_24-25.csv"
# -------------------------------------------------------

# Step 1: Load dataset
df = pd.read_csv(CSV_FILE_PATH)

# Step 2: Select required input fields
input_fields = ["length", "cost", "state", "year", "district", "work_status"]
input_values = [df.iloc[0][col] for col in input_fields]

# Step 3: Generate IAM Bearer Token
print("🔑 Getting IAM token...")
token_response = requests.post(
    url='https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"},
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

if token_response.status_code != 200:
    print("❌ Failed to get token:", token_response.text)
    exit(1)

access_token = token_response.json()["access_token"]
print("✅ Token retrieved.")

# Step 4: Prepare payload for prediction
payload = {
    "input_data": [{
        "fields": input_fields,
        "values": [input_values]
    }]
}

# Step 5: Send prediction request
print("📡 Sending prediction request to IBM Cloud ML...")
prediction_response = requests.post(
    url=DEPLOYMENT_URL,
    json=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
)

# Step 6: Display result
if prediction_response.status_code == 200:
    result = prediction_response.json()
    print("✅ Prediction received:")
    print(json.dumps(result, indent=2))
else:
    print("❌ Prediction failed:", prediction_response.text)
