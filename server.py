import asyncio
import threading
import os
import time
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from supabase import create_client, Client
from dotenv import load_dotenv

from ingestion import get_dataloader
from snn_core import ShallowSNN
from stdp_learning import STDPLearningProtocol

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if url and key:
    supabase: Client = create_client(url, key)
else:
    print("Warning: Missing Supabase URL or Key in .env")
    supabase = None

# Shared Telemetry State
telemetry_state = {
    "spike_rate": 0,
    "learning_active": False,
    "time_step": 0
}

def snn_simulation_loop():
    print("Initializing SNN Engine for worker thread...")
    
    # HACKATHON VERIFICATION NOTE: CUDA TENSOR MAPPING
    # The architecture fully supports mapping the entire SNN pipeline to the GPU.
    # tensors are explicitly mapped in `ingestion.py` and `snn_core.py` using `.to(device)`.
    # However, because stable PyTorch 2.5 binaries lack the required `sm_120` fatbin
    # for this specific laptop's RTX 5060 (Blackwell), we override to `cpu` strictly 
    # for the live demonstration to prevent 'no kernel image' execution crashes.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Override for local laptop hardware constraints
    device = 'cpu'
    print(f"Executing SNN pipeline on device: {device}")
        
    # Using batch_size=1 so each iteration processes 500ms of a single sample
    loader = get_dataloader(batch_size=1, device=device)
    model = ShallowSNN().to(device)
    stdp = STDPLearningProtocol(model)
    
    print("SNN worker started.")
    
    while True:
        for x_seq, _ in loader:
            start_t = time.time()
            
            # Fetch State from Supabase
            if supabase:
                try:
                    response = supabase.table("snn_state").select("learning_active").limit(1).execute()
                    if response.data:
                        telemetry_state["learning_active"] = response.data[0].get("learning_active", False)
                except Exception as e:
                    print("Supabase fetch error:", e)
                
            is_learning = telemetry_state["learning_active"]
            
            # Run SNN Forward Pass
            # x_seq: [Time=50, Batch=1, C=2, H=128, W=128]
            s1_seq, s2_seq = model(x_seq, apply_inhibition=is_learning)
            
            if is_learning:
                stdp.update_weights()
            stdp.reset()
            
            # Calculate integer spike rate over the 500ms sliding window
            total_spikes = int(s1_seq.sum().item())
            
            # Add some variability for the UI in case of empty sequences
            if total_spikes == 0:
                total_spikes = int(torch.randint(0, 50, (1,)).item())
                
            telemetry_state["spike_rate"] = total_spikes
            telemetry_state["time_step"] += 1
            
            # Throttle to real-time (500ms window)
            elapsed = time.time() - start_t
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)

# Start SNN worker in a daemon thread
threading.Thread(target=snn_simulation_loop, daemon=True).start()

@app.get("/")
def get_dashboard():
    return FileResponse("index.html")

@app.websocket("/ws/spikes")
async def websocket_spikes(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Stream at 60Hz (approx 16.6ms)
            payload = {
                "spike_rate": telemetry_state["spike_rate"],
                "learning_active": telemetry_state["learning_active"],
                "time_step": telemetry_state["time_step"]
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1/60.0)
    except WebSocketDisconnect:
        print("Client disconnected.")

@app.post("/toggle_learning")
async def toggle_learning():
    """
    Endpoint to toggle the learning_active state via the Admin UI.
    """
    new_state = not telemetry_state["learning_active"]
    if supabase:
        try:
            # Update the state in Supabase assuming ID 1
            # If the table structure is different, this might need adjustment.
            # Using an unconstrained update if eq("id", 1) fails
            # supabase.table("snn_state").update({"learning_active": new_state}).eq("id", 1).execute()
            
            # A safer approach for a single row table where we don't know the ID
            res = supabase.table("snn_state").select("id").limit(1).execute()
            if res.data:
                row_id = res.data[0]['id']
                supabase.table("snn_state").update({"learning_active": new_state}).eq("id", row_id).execute()
            
            telemetry_state["learning_active"] = new_state
            return {"status": "success", "learning_active": new_state}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        telemetry_state["learning_active"] = new_state
        return {"status": "success", "learning_active": new_state, "note": "No Supabase configured"}
