# 🧠 Software-Simulated Neuromorphic SNN Pipeline

A real-time, hardware-accelerated Spiking Neural Network (SNN) pipeline designed to process asynchronous neuromorphic event data (DVS) with a biologically plausible Spike-Timing-Dependent Plasticity (STDP) learning protocol. 

Built with **PyTorch**, **SpikingJelly**, **FastAPI**, and **Supabase**, featuring a live 60Hz telemetry dashboard to monitor network spiking behavior and lateral inhibition regulation.

## ✨ Key Features
* **Neuromorphic Data Ingestion**: Parses the IBM DVSGesture dataset into dense 10ms temporal tensors.
* **SNN Core Engine**: Multi-step Leaky Integrate-and-Fire (LIF) network optimized for tensor-level processing.
* **Biologically Plausible STDP**: Features real-time unsupervised learning with mathematical lateral inhibition to prevent winner-take-all weight collapse.
* **Decoupled Telemetry**: FastAPI backend streaming live network spike densities at 60Hz over WebSockets.
* **Remote State Mutation**: Supabase integration allowing remote administrative toggles to inject the STDP learning protocol dynamically during runtime.

## 🚀 How to Run Locally

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shaswat1110/neuromorphic-poc.git
cd neuromorphic-poc

# Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Supabase (Optional)
**Note for Judges:** The architecture is designed to automatically fall back to a local, in-memory state if no database is detected. You do **NOT** need to set up Supabase to test this project. It will run perfectly out of the box!

*(Optional)* If you wish to test the remote state mutation, create a `.env` file in the root directory:
```env
SUPABASE_URL="your-supabase-url"
SUPABASE_KEY="your-anon-key"
```
Ensure your Supabase project has a table named `snn_state` with a boolean column `learning_active`.

### 3. Launch the Pipeline
```bash
python -m uvicorn server:app --reload
```

### 4. View the Telemetry Dashboard
Open your browser and navigate to:
👉 **http://127.0.0.1:8000**

You will see the 60Hz real-time spike density of the network. Click **Enable STDP** to inject the learning protocol and observe the lateral inhibition stabilizing the firing rate.

## 🐳 Docker Deployment (Optional)
If you wish to deploy this to the cloud for asynchronous judging:
```bash
docker build -t neuromorphic-snn .
docker run -p 8000:8000 neuromorphic-snn
```

## 🛠️ Tech Stack
* **AI/ML**: PyTorch, SpikingJelly, Tonic
* **Backend**: FastAPI, Uvicorn, Python, WebSockets
* **Database**: Supabase
* **Frontend**: HTML5, Vanilla JavaScript, Chart.js, CSS3 (Glassmorphism/Neon UI)
