# Secure Distributed ML Architecture (CipherSight)

This project implements a privacy-preserving machine learning architecture split across an Android Client, a Python Intermediary Server, and a Trusted Model Owner. It utilizes **Deterministic Cryptographic Noise**, **Additive Secret Sharing (SMPC)**, and **Cosine Similarity Gradient Filtering** to protect both client data and model parameters.

## Prerequisites & Required Software

### 1. Python Backend (Servers)
Ensure you have Python 3.9 or higher installed on your machine.
The servers require the following libraries to process tensors and calculate cosine similarity:
*   `numpy`
*   `scipy`

### 2. Android Client
*   **IDE:** Android Studio (Ladybug or newer recommended)
*   **SDK:** Target SDK 37 (Min SDK 24)
*   **Libraries (Handled via Gradle):** 
    *   TensorFlow Lite (2.16.1)
    *   Ktor Client (2.3.5)
    *   Kotlin Coroutines (1.7.3)

---

## Installation & Setup Instructions

### Step 1: Set Up the Python Environment
1. Open a terminal and navigate to the project directory containing the Python scripts.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install numpy scipy
   ```

### Step 2: Configure the Network (Important!)
For the Android app to communicate with the Python servers, they must be on the same local network (Wi-Fi), and the app needs to know your computer's local IP address.
1. Find your computer's local IPv4 address (e.g., `192.168.x.x`).
    * *Windows:* Run `ipconfig` in the Command Prompt.
    * *Mac/Linux:* Run `ifconfig` or `ip a` in the terminal.
2. Open the Android project in Android Studio.
3. Navigate to `app/src/main/java/com/example/CypherSight/MainActivity.kt`.
4. Update the server URLs at the top of the `RealInferenceScreen` function with your computer's actual IP address:
   ```kotlin
   val thirdPartyUrl = "http://YOUR_LOCAL_IP:8000/process_features"
   val trustedOwnerUrl = "http://YOUR_LOCAL_IP:8001/register_share"
   ```

---

## How to Run the Project

You must start the backend servers before attempting to process an image on the Android app.

**1. Start the Trusted Owner Server:**
Open a terminal, navigate to the folder containing `trusted_owner.py`, and run:
```bash
python trusted_owner.py
```
*(This server will listen on port 8001 and wait for Share B / the initialization vector).*

**2. Start the Intermediary Server:**
Open a second, separate terminal, navigate to the folder containing `server.py`, and run:
```bash
python server.py
```
*(This server will listen on port 8000 and wait for Share A / the noisy latent vector).*

**3. Run the Android Application:**
*   Either run the app directly from Android Studio onto a physical device/emulator via USB debugging, or install the provided `app-debug.apk`.
*   Ensure the Android device is connected to the **same Wi-Fi router** as the computer running the Python servers.
*   Open the app, select an image from the gallery or take a photo, and tap **"Process via Servers"**.

## Demonstration Guide
When you press the process button on the app, watch the two terminal windows running the Python scripts. 
*   The `server.py` terminal will show the incoming noise-injected tensor arriving (demonstrating the Cryptographic Noise and SMPC). 
*   The `trusted_owner.py` terminal will show the reconstruction logic and Cosine Similarity Gradient Filtering validation, proving the architecture's security mechanisms are actively defending the system.
