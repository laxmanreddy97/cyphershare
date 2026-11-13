from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import requests
import uvicorn
import urllib.request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize the rate limiter to track requests by the client's IP address
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# Register the limiter with the FastAPI app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

try:
    bulk_model = tf.keras.models.load_model('third_party_bulk.h5')
except Exception as e:
    print(f"Error loading model: {e}. Ensure you ran the updated splitter.py first.")

TRUSTED_OWNER_URL = "http://127.0.0.1:8001/process_critical"

labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels = []
try:
    with urllib.request.urlopen(labels_url) as response:
        labels = response.read().decode('utf-8').splitlines()[1:] 
except Exception as e:
    print(f"Warning: Could not download labels: {e}")

# ... (Imports and model loading remain the same) ...

class ClientData(BaseModel):
    request_id: str
    features: list[float]
    # IV has been removed. The TP Server is now 100% zero-knowledge.

@app.post("/process_features")
@limiter.limit("10/hour")
def process_features(request: Request, data: ClientData):
    expected_client_shape = (1, 14, 14, 256)
    
    try:
        client_noisy_tensor = np.array(data.features, dtype=np.float32).reshape(expected_client_shape)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Shape mismatch: {ve}")
    
    owner_payload = {
        "request_id": data.request_id,
        "features": data.features,
        "shape": list(expected_client_shape)
    }
    
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    custom_headers = {"X-Forwarded-For": client_ip}
    
    try:
        response = requests.post(TRUSTED_OWNER_URL, json=owner_payload, headers=custom_headers)
        response.raise_for_status()
        owner_data = response.json()
        # ... (Rest of the inference block remains identical) ...        
        owner_tensor = np.array(owner_data["critical_features"], dtype=np.float32).reshape(owner_data["critical_shape"])
        noise_offset_tensor = np.array(owner_data["bulk_noise_offset"], dtype=np.float32).reshape(owner_data["bulk_shape"])
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Trusted Owner communication failed: {e}")
    except KeyError as e:
        raise HTTPException(status_code=502, detail="Invalid response format from Trusted Owner")

    try:
        predictions = bulk_model.predict([client_noisy_tensor, owner_tensor, noise_offset_tensor], verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    max_index = int(np.argmax(predictions))
    confidence = float(predictions[0][max_index])
    
    class_name = f"Class {max_index}"
    if labels and max_index < len(labels):
        class_name = labels[max_index]
    
    return {
        "status": "success",
        "prediction": class_name,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)