import os
import torch
from fastapi import APIRouter

from core import config
from core.tokenizer import Tokenizer
from core.model import TransformersDecoder

from serving.schemas import UserInput, ModelOutput

router = APIRouter(prefix="/chat", tags=["chat"])

tokenizer = Tokenizer()
transformer_model = TransformersDecoder(vocab_size=config.VOCAB_SIZE,
                                            d_model=config.D_MODEL,
                                            num_heads=config.N_HEADS,
                                            hidden_layer_dim=config.HIDDEN_LAYER_DIM,
                                            num_blocks=config.N_BLOCKS,
                                            max_seq_len=config.SEQ_LEN)

checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pth")

assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

model_checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
transformer_model.load_state_dict(model_checkpoint['model'])

@router.post("/model_inference", response_model=ModelOutput)
def generate(request: UserInput):
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
