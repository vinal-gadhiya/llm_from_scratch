import torch

from core import config
from core.tokenizer import Tokenizer
from core.model import TransformersDecoder

tokenizer = Tokenizer()
transformer_decoder = TransformersDecoder(vocab_size=config.VOCAB_SIZE,
                                            d_model=config.D_MODEL,
                                            num_heads=config.N_HEADS,
                                            hidden_layer_dim=config.HIDDEN_LAYER_DIM,
                                            num_blocks=config.N_BLOCKS,
                                            max_seq_len=config.SEQ_LEN)

def generate(input_text):
    transformer_decoder.eval()
    transformer_decoder.reset_cache()
    tokens_list = tokenizer.encode(input_text, config.VOCAB_PATH, config.MERGE_PATH)
    model_pred = transformer_decoder(torch.tensor(tokens_list).to(config.DEVICE))[-1]
    probs = torch.softmax(model_pred, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    input_token = sampled_index.item()
    tokens_list.append(input_token)

    for _ in range(100):
        model_pred = transformer_decoder(torch.tensor([input_token]).to(config.DEVICE))
        probs = torch.softmax(model_pred, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        input_token = sampled_index.item()
        tokens_list.append(sampled_index.item())
    decoded_output = tokenizer.decode(tokens_list, config.VOCAB_PATH)
    return decoded_output

print(generate("Hello there!!"))