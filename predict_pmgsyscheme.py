import pandas as pd
import requests
import json

# ---------------------- CONFIGURATION ----------------------
API_KEY = "ApiKey-699eac6f-b1c9-4407-ac7b-1e08242972bd"
DEPLOYMENT_URL = "https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/6a329373-216e-4b77-bb42-a7eb799a997a/predictions?version=2021-05-01"
CSV_FILE_PATH = "DistrictswiseCR_AEdataf_24-25.csv"  # Make sure this is in the same directory
# -----------------------------------------------------------

# Step 1: Load dataset
df = pd.read_csv(CSV_FILE_PATH)

# Step 2: Preview column names (optional)
print("🧾 Columns:", df.columns.tolist())

# Step 3: Select a sample row for prediction
sample_row = df.iloc[0]

# You must update this list based on the actual fields your model expects
input_fields = ["length", "cost", "state", "year", "district", "work_status"]

# Extract values from the sample
input_values = [sample_row[col] for col in input_fields]

# Step 4: Get Bearer Token
print("🔑 Getting Bearer Token...")
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"},
    headers={"Content-Type": "application/x-www-form-urlencoded"}
)

if token_response.status_code != 200:
    print("❌ Error getting token:", token_response.text)
    exit()

access_token = token_response.json()["access_token"]
print("✅ Token received.")

# Step 5: Prepare payload for prediction
payload = {
    "input_data": [{
        "fields": input_fields,
        "values": [input_values]
    }]
}

# Step 6: Send prediction request
print("📤 Sending prediction request...")
response = requests.post(
    DEPLOYMENT_URL,
    json=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
)

# Step 7: Output prediction
if response.status_code == 200:
    print("✅ Prediction Result:")
    print(json.dumps(response.json(), indent=2))
else:
    print("❌ Prediction failed:", response.text)
