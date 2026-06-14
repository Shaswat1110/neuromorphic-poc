# 🧠 Software-Simulated Neuromorphic SNN Architecture

This document outlines the architecture, data flow, and technology stack of our real-time Spiking Neural Network (SNN) pipeline. It bridges the gap between biological plausibility and modern hardware-accelerated deep learning.

---

## 🏗️ System Architecture Diagram

```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#0f172a,stroke:#00f3ff,stroke-width:2px,color:#fff
    classDef backend fill:#1e1b4b,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef neuromorphic fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff
    classDef database fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#fff

    subgraph Data Ingestion Layer
        DVS[IBM DVSGesture Dataset<br/>Asynchronous Events]:::neuromorphic
        Tonic[Tonic Transformer<br/>Batches to 10ms Tensors]:::neuromorphic
    end

    subgraph PyTorch SNN Engine
        LIF[Leaky Integrate-and-Fire <br/>LIFNode Network]:::backend
        STDP[STDP Learning Protocol<br/>Biological Weight Updates]:::backend
        LatInhib[Mathematical <br/>Lateral Inhibition]:::backend
    end

    subgraph FastAPI Backend
        Daemon[Background Thread<br/>SNN Execution Loop]:::backend
        WS[WebSocket Endpoint<br/>60Hz Telemetry Stream]:::backend
        API[REST API<br/>/toggle_learning]:::backend
    end

    subgraph Frontend Dashboard
        UI[Glassmorphism UI]:::frontend
        Chart[Chart.js Live Graph<br/>Spike Rate Density]:::frontend
    end

    subgraph Remote State
        Supa[(Supabase DB<br/>snn_state table)]:::database
    end

    %% Flow connections
    DVS -- Raw Data --> Tonic
    Tonic -- "[T, B, C, H, W]" --> LIF
    LIF <--> STDP
    LIF <--> LatInhib
    
    LIF -- Total Spikes/500ms --> Daemon
    Daemon -- Payload --> WS
    WS -- JSON Stream --> Chart
    
    UI -- Toggle Request --> API
    API -- Update --> Supa
    Supa -- Poll Status --> Daemon
    Daemon -- Enable/Disable --> STDP
```

---

## ⚙️ Component Breakdown

### 1. Data Ingestion (`ingestion.py`)
*   **Technology:** `tonic`, `torch.utils.data.DataLoader`
*   **Role:** Replaces standard video ingestion. It streams the IBM DVSGesture dataset, which consists of event-based neuromorphic camera recordings (changes in brightness recorded at the microsecond level).
*   **Processing:** Converts the sparse, asynchronous events into dense temporal tensors `[Time, Batch, Channels, Height, Width]` chopped into 500ms sliding windows for the PyTorch network.

### 2. SNN Core Engine (`snn_core.py`)
*   **Technology:** `PyTorch` (CUDA-enabled), `SpikingJelly`
*   **Role:** The "Brain". A shallow, step-mode Spiking Neural Network utilizing `LIFNode` (Leaky Integrate-and-Fire) neurons. 
*   **Feature:** Implements mathematical **Lateral Inhibition**. When a neuron fires, it applies an explicit negative voltage bias to neighboring neurons, preventing chaotic "winner-take-all" firing collapse.

### 3. Biological Learning (`stdp_learning.py`)
*   **Technology:** `spikingjelly.activation_based.learning.STDPLearner`
*   **Role:** Implements unsupervised **Spike-Timing-Dependent Plasticity**. 
*   **Execution:** Instead of backpropagation (which relies on standard math derivatives), STDP adjusts synapse weights biologically based on the precise temporal differences between pre-synaptic and post-synaptic spikes.

### 4. Server & Telemetry (`server.py`)
*   **Technology:** `FastAPI`, `WebSockets`, `Python Threading`
*   **Role:** The orchestration layer. The PyTorch SNN runs infinitely inside a daemon background thread. 
*   **Decoupling:** As the network computes, it drops its output (total spikes) into a shared dictionary. The WebSocket endpoint reads this dictionary and streams the spike density to the frontend at a buttery smooth **60 frames per second**.

### 5. Remote State Mutation (Supabase)
*   **Technology:** `supabase-py`
*   **Role:** Allows remote administrative control. The backend polls the Supabase `snn_state` table for the `learning_active` flag. This allows us to inject the STDP protocol dynamically into the running PyTorch simulation remotely without restarting the server.

### 6. Frontend UI (`index.html`)
*   **Technology:** Vanilla JS, HTML5, `Chart.js`
*   **Role:** A highly responsive, neon-aesthetic dashboard that ingests the 60Hz WebSocket stream and visualizes the network's internal spike rate density, proving that the lateral inhibition actively stabilizes the neural firing pattern.
