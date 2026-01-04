import torch
import os
import torch.optim as optim

from core import config
from core.tokenizer import Tokenizer
from core.model import TransformersDecoder
from training.trainer import Trainer

with open(config.INPUT_FILE_PATH, "r") as f:
    shakespere_text = f.read()


tokenizer = Tokenizer()

if not os.path.exists(config.VOCAB_PATH):
    tokenizer.save_vocab(shakespere_text, n=config.N_TOKENIZER_TRAIN_STEPS)


transformer_decoder = TransformersDecoder(vocab_size=config.VOCAB_SIZE,
                                            d_model=config.D_MODEL,
                                            num_heads=config.N_HEADS,
                                            hidden_layer_dim=config.HIDDEN_LAYER_DIM,
                                            num_blocks=config.N_BLOCKS)


optimizer = optim.Adam(transformer_decoder.parameters(), lr=config.LR)
loss_fn = torch.nn.CrossEntropyLoss()

tokenized_text = tokenizer.encode(shakespere_text, vocab_path=config.VOCAB_PATH, merge_path=config.MERGE_PATH)

trainer = Trainer(model=transformer_decoder, config=config, optimizer=optimizer, criterion=loss_fn)

trainer.train_iterations(tokenized_text=tokenized_text, num_iterations=config.N_TRAINING_ITERATIONS)
