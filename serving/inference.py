import os
import torch
from fastapi import APIRouter, HTTPException

from core import config
from core.tokenizer import Tokenizer
from core.model import TransformersDecoder

from serving.state import model_state
from serving.schemas import UserInput, ModelOutput

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/model_inference", response_model=ModelOutput)
def generate(request: UserInput):
    if not model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {model_state["error"]}"
        )
    
    tokenizer = model_state["tokenizer"]
    transformer_model = model_state["model"]

    transformer_model.eval()
    transformer_model.reset_cache()
    input_text = request.user_input
    tokens_list = tokenizer.encode(input_text, config.VOCAB_PATH, config.MERGE_PATH)
    model_pred = transformer_model(torch.tensor(tokens_list).to(config.DEVICE))[-1]
    probs = torch.softmax(model_pred, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    input_token = sampled_index.item()
    tokens_list.append(input_token)

    for _ in range(100):
        model_pred = transformer_model(torch.tensor([input_token]).to(config.DEVICE))
        probs = torch.softmax(model_pred, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        input_token = sampled_index.item()
        tokens_list.append(sampled_index.item())
    decoded_output = tokenizer.decode(tokens_list, config.VOCAB_PATH)
    return ModelOutput(model_output=decoded_output)
