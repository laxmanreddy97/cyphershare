from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import uvicorn
import base64
import struct
from Crypto.Cipher import AES
from Crypto.Util import Counter
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

def get_real_ip(request: Request):
    return request.headers.get("X-Forwarded-For", request.client.host)

limiter = Limiter(key_func=get_real_ip)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

try:
    owner_model = tf.keras.models.load_model('trusted_owner.h5')
    full_bulk_model = tf.keras.models.load_model('third_party_bulk.h5')
    bulk_pw_6_layer = full_bulk_model.get_layer('bulk_pw_6')

    noise_input = tf.keras.Input(shape=(14, 14, 256))
    noise_output = bulk_pw_6_layer(noise_input)
    bulk_noise_model = tf.keras.Model(inputs=noise_input, outputs=noise_output)
except Exception as e:
    print(f"Error loading models: {e}")

SHARED_KEY = b"CryptoAI_SecretKey_256bit_32Byte"

active_shares = {}

# ==========================================
# SECURITY: Cosine Similarity State
# ==========================================
BASELINE_FEATURES = None
MOMENTUM = 0.99          # 99% past history, 1% new tensor
SIMILARITY_THRESHOLD = 0.70  # Reject anything below 70% structural similarity

def calculate_cosine_similarity(vec_a, vec_b):
    a = vec_a.flatten()
    b = vec_b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class ShareData(BaseModel):
    request_id: str
    iv: str

class LatentData(BaseModel):
    request_id: str
    features: list[float]
    shape: list[int]

@app.post("/register_share")
def register_share(data: ShareData):
    active_shares[data.request_id] = data.iv
    return {"status": "share_secured"}

@app.post("/process_critical")
@limiter.limit("10/hour") 
def process_critical(request: Request, data: LatentData):
    global BASELINE_FEATURES
    
    iv_base64 = active_shares.pop(data.request_id, None)
    if not iv_base64:
        raise HTTPException(status_code=404, detail="Share B (IV) not found or expired.")

    try:
        iv_bytes = base64.b64decode(iv_base64)
        ctr = Counter.new(128, initial_value=int.from_bytes(iv_bytes, byteorder='big'))
        cipher = AES.new(SHARED_KEY, AES.MODE_CTR, counter=ctr)
        
        total_elements = int(np.prod(data.shape))
        keystream = cipher.encrypt(b'\x00' * (total_elements * 4))
        
        noise_floats = []
        for i in range(total_elements):
            rand_int = struct.unpack('<i', keystream[i*4:(i+1)*4])[0]
            noise_float = (rand_int / 2147483647.0) * 2.0
            noise_floats.append(noise_float)
            
        noise_matrix = np.array(noise_floats, dtype=np.float32).reshape(data.shape)
        
        noisy_tensor = np.array(data.features, dtype=np.float32).reshape(data.shape)
        clean_tensor = noisy_tensor - noise_matrix
        
        # ==========================================
        # ADVANCED SECURITY: Adversarial Check
        # ==========================================
        if BASELINE_FEATURES is None:
            # Initialize baseline (In production, pre-warm this with a known good image)
            BASELINE_FEATURES = clean_tensor.copy()
        else:
            similarity = calculate_cosine_similarity(clean_tensor, BASELINE_FEATURES)
            print(f"Request {data.request_id} Similarity: {similarity:.4f}")
            
            if similarity < SIMILARITY_THRESHOLD:
                print(f"SECURITY ALERT: Blocked adversarial payload. Similarity: {similarity:.4f}")
                raise HTTPException(status_code=403, detail="Payload rejected by structural security filter.")
            
            # Update the baseline with the new, benign tensor
            BASELINE_FEATURES = (MOMENTUM * BASELINE_FEATURES) + ((1.0 - MOMENTUM) * clean_tensor)
        
        # Proceed with normal inference
        critical_output = owner_model.predict(clean_tensor, verbose=0)
        bulk_noise_offset = bulk_noise_model.predict(noise_matrix, verbose=0)
        
        return {
            "critical_features": critical_output.flatten().tolist(),
            "critical_shape": list(critical_output.shape),
            "bulk_noise_offset": bulk_noise_offset.flatten().tolist(),
            "bulk_shape": list(bulk_noise_offset.shape)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)