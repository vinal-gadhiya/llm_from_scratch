import os
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager

from core import config
from core.tokenizer import Tokenizer
from core.model import TransformersDecoder

from serving import inference

model_state = {
    "tokenizer": None,
    "model": None,
    "loaded": False,
    "error": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    try:

        model_state["tokenizer"] = Tokenizer()
        model_state["model"] = TransformersDecoder(vocab_size=config.VOCAB_SIZE,
                                                    d_model=config.D_MODEL,
                                                    num_heads=config.N_HEADS,
                                                    hidden_layer_dim=config.HIDDEN_LAYER_DIM,
                                                    num_blocks=config.N_BLOCKS,
                                                    max_seq_len=config.SEQ_LEN)

        # checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pth")
        checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/latest.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model_state["model"].load_state_dict(model_checkpoint['model'])
        model_state["loaded"] = True
        model_state["error"] = None
    
    except Exception as e:
        model_state["loaded"] = False
        model_state["error"] = str(e)
    
    yield
    print("Shutting down...")
    model_state["tokenizer"] = None
    model_state["model"] = None
    model_state["loaded"] = False

app = FastAPI(title="LLM serving app", lifespan=lifespan)

app.include_router(inference.router)