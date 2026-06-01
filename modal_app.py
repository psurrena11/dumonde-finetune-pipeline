"""
Deploy the duMonde finetune on Modal as an OpenAI-compatible API.

Serves the Q8_0 GGUF through Ollama on a GPU container. The model weights
live in a Modal Volume (uploaded once with `modal volume put`), and the model
is registered with Ollama on first cold start, then cached in the Volume so
later cold starts skip the import.

Auth: callers (the duMonde backend) must send `Authorization: Bearer <API_KEY>`,
where API_KEY comes from the `dumonde-api-key` Modal Secret.

Persona: the Modelfile bakes a default SYSTEM persona. Callers may override it
per request by including a `system` message in the chat payload.

Setup (run once):
    modal volume create dumonde-models
    modal volume put dumonde-models gemma4-bws-Q8_0.gguf /gemma4-bws-Q8_0.gguf
    modal secret create dumonde-api-key API_KEY=$(openssl rand -hex 24)

Deploy:
    modal deploy modal_app.py
"""

import os
import subprocess
import time

import modal

MODEL_NAME = "dumonde"
MODELS_DIR = "/models"          # Volume mount; holds the GGUF
GGUF_FILE = "gemma4-bws-Q8_0.gguf"
OLLAMA_PORT = 11434

# Persist Ollama's model store inside the Volume so the GGUF import (~5GB copy)
# happens once, not on every cold start.
OLLAMA_MODELS_DIR = "/models/ollama"

# Ollama 0.24's `create` resolves a relative FROM against the Modelfile's
# directory, so we write the Modelfile next to the GGUF and reference it
# relatively. (Absolute paths and stdin `-f -` both fail with "no Modelfile or
# safetensors files found" on this version.)
MODELFILE = f"""\
FROM ./{GGUF_FILE}

SYSTEM \"\"\"You are a lean left black christian from the northeast usa. You speak in natural, conversational first-person. Short sentences but you also give context. No bullet points. No academic framing. You don't explain yourself, you just talk. You don't shy away from an answer, you are direct even if it's harsh. You do not need to remain neutral or objective. Your truth is important to you.\"\"\"

TEMPLATE \"\"\"<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
\"\"\"

PARAMETER stop "<end_of_turn>"
PARAMETER stop "Thinking..."
PARAMETER stop "Here's a plan"
PARAMETER stop "*(Self-Correction"
PARAMETER temperature 0.8
PARAMETER top_p 0.9
"""

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("fastapi[standard]==0.115.4", "httpx==0.27.2")
    .env({"OLLAMA_MODELS": OLLAMA_MODELS_DIR})
)

app = modal.App("dumonde")
volume = modal.Volume.from_name("dumonde-models")


def _wait_for_ollama(timeout: int = 60):
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/version", timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Ollama server did not become ready in time")


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("dumonde-api-key")],
    scaledown_window=1800,  # keep warm 30 min after last request
    timeout=600,
)
@modal.concurrent(max_inputs=10)
class Server:
    @modal.enter()
    def start(self):
        os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)
        # Launch the Ollama daemon in the background.
        self.proc = subprocess.Popen(
            ["ollama", "serve"],
            env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}"},
        )
        _wait_for_ollama()

        # Register the model from the GGUF if not already in the Volume cache.
        existing = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        ).stdout
        if MODEL_NAME not in existing:
            print(f"Importing {MODEL_NAME} from {MODELS_DIR}/{GGUF_FILE} (first run)...")
            modelfile_path = os.path.join(MODELS_DIR, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(MODELFILE)
            # Run from the GGUF's directory so the relative FROM resolves.
            result = subprocess.run(
                ["ollama", "create", MODEL_NAME, "-f", "Modelfile"],
                cwd=MODELS_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ollama create failed (exit {result.returncode}):\n"
                    f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
            volume.commit()  # persist the imported model for future cold starts
        else:
            print(f"{MODEL_NAME} already present in volume cache.")

    @modal.asgi_app()
    def api(self):
        import httpx
        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        web = FastAPI(title="duMonde Inference API")
        api_key = os.environ["API_KEY"]

        def check_auth(authorization: str | None):
            if authorization != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        @web.get("/health")
        def health():
            return {"status": "ok", "model": MODEL_NAME}

        @web.get("/v1/models")
        async def models(authorization: str | None = Header(default=None)):
            check_auth(authorization)
            url = f"http://127.0.0.1:{OLLAMA_PORT}/v1/models"
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(url)
                return JSONResponse(status_code=r.status_code, content=r.json())

        @web.post("/v1/chat/completions")
        async def chat(request: Request, authorization: str | None = Header(default=None)):
            check_auth(authorization)
            payload = await request.json()
            payload.setdefault("model", MODEL_NAME)
            url = f"http://127.0.0.1:{OLLAMA_PORT}/v1/chat/completions"

            async with httpx.AsyncClient(timeout=300) as client:
                if payload.get("stream"):
                    async def gen():
                        async with client.stream("POST", url, json=payload) as r:
                            async for chunk in r.aiter_bytes():
                                yield chunk
                    return StreamingResponse(gen(), media_type="text/event-stream")
                r = await client.post(url, json=payload)
                return JSONResponse(status_code=r.status_code, content=r.json())

        return web
