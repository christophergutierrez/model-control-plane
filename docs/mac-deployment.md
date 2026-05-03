# Deploying to Mac

This document lists everything needed to run the model control plane on a Mac. It assumes the Mac has Apple Silicon (M1/M2/M3/M4) and macOS 13+.

## Source Machine

- Hostname: `edgexpert-782a`
- IP: `192.168.2.103`
- Registry root: `/home/chris/models`
- Repo: `/home/chris/git_home/model-control-plane`

## What to Copy

### 1. Registry (routes + adapters) — ~1.5 GB total

The entire `/home/chris/models` directory. Contains:

- **Route configs** (19 routes): `/home/chris/models/routes/`
- **LoRA adapters** (18 adapters): `/home/chris/models/adapters/`

Each adapter is a PEFT LoRA (rank 16, safetensors format) trained on `Qwen/Qwen2.5-Coder-1.5B-Instruct`. The weights are hardware-independent — they work on any machine that can load the base model.

### 2. This repo

```
git pull
```

The repo contains all orchestrator code, router, launch scripts, and tools.

## Files NOT Needed

- The base model (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) — downloads automatically from HuggingFace on first run (~3 GB)
- The vLLM venv at `/home/chris/vllm-install/.vllm` — must be installed fresh on the Mac
- The embedding model (`all-MiniLM-L6-v2`) — downloads automatically on first run (~80 MB)
- The numpy embedding cache at `tools/.cache/` — regenerates automatically

## rsync Commands (run on Mac)

Pick a location for the registry on the Mac. These examples use `~/models`:

```bash
# Copy the full registry (routes + adapters)
rsync -avz --progress chris@192.168.2.103:/home/chris/models/ ~/models/
```

That's the only large transfer. The repo itself comes via git.

## Setup Instructions (for AI on Mac)

### Step 1: Install Python dependencies

Create a venv and install the required packages:

```bash
python3 -m venv ~/.venvs/mcp
source ~/.venvs/mcp/bin/activate
pip install vllm torch transformers peft sentence-transformers
```

If vLLM doesn't install on Mac (it may not support MPS yet), use this fallback stack instead:

```bash
pip install torch transformers peft accelerate sentence-transformers
```

### Step 2: Verify the base model downloads

```bash
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')"
```

This downloads ~3 GB on first run. Subsequent runs use the HuggingFace cache.

### Step 3: Update adapter paths in the registry

Every adapter's `manifest.json` has an absolute `adapter_path` pointing to `/home/chris/models/...`. These need to be updated to match the Mac's path.

Run from the repo root:

```bash
find ~/models -name "manifest.json" -exec sed -i '' 's|/home/chris/models|/Users/YOUR_USERNAME/models|g' {} +
```

Also update the `route.json` files if they contain absolute paths:

```bash
find ~/models -name "route.json" -exec sed -i '' 's|/home/chris/models|/Users/YOUR_USERNAME/models|g' {} +
```

### Step 4: Test the registry resolves

```bash
cd /path/to/model-control-plane
python3 tools/resolve_route.py ~/models videoamp/api/programs
```

Should print the route key, version, base model, and adapter path (now pointing to the Mac path).

### Step 5: Choose a serving backend

#### Option A: vLLM (if it installs on Mac)

```bash
python3 tools/serve_vllm.py ~/models videoamp/api/programs \
    --role responder --port 8000 --dtype float16 \
    -- --enforce-eager
```

Note: use `float16` not `bfloat16` — older MPS backends may not support bfloat16.

#### Option B: transformers + PEFT (no vLLM needed)

If vLLM doesn't work on Mac, you can load adapters directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "~/models/adapters/Qwen/Qwen2.5-Coder-1.5B-Instruct/videoamp/api/programs/responder/v1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
```

This won't give you the multi-LoRA OpenAI-compatible server, but it proves the weights work.

#### Option C: MLX (Apple-native, fastest on Mac)

```bash
pip install mlx-lm
```

MLX supports LoRA adapters natively and runs well on Apple Silicon. This is likely the best Mac experience but requires converting adapters to MLX format.

### Step 6: Launch the orchestrator

Once a serving backend is running on port 8000:

```bash
python3 tools/orchestrate.py ~/models --serve --serve-port 8080
```

Open `http://localhost:8080/` in a browser to see the dashboard.

### Step 7: Verify end to end

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/routes
curl -s http://localhost:8080 -d '{"prompt": "list all programs"}'
```

## Adapter Details

- Format: PEFT LoRA (safetensors)
- Rank: 16
- Alpha: 32
- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Qwen2ForCausalLM)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Size per adapter: ~82 MB
- Total adapters: 18 (17 responder + 1 router)
- Total adapter size: ~1.5 GB
