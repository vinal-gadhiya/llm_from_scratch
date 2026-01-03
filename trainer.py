import os
import random

import torch

class Trainer:
    def __init__(self, model, config, optimizer, criterion):
        self.config = config
        self.model = model.to(self.config.DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.log_interval = 100

    def train_step(self, input_seq, target_seq):
        input_seq = input_seq.to(self.config.DEVICE)
        target_seq = target_seq.to(self.config.DEVICE)
        logits = self.model(input_seq)
        loss = self.criterion(
            logits.view(-1, self.config.VOCAB_SIZE),
            target_seq.view(-1)
        )
        self.optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_iterations(self, tokenized_text, num_iterations):
        self.model.train()

        if not torch.is_tensor(tokenized_text):
            tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
        tokenized_text = tokenized_text.to(self.config.DEVICE)

        running_loss = 0.0

        for step in range(1, num_iterations + 1):
            start_idx = random.randint(0, len(tokenized_text) - self.config.SEQ_LEN - 1)
            input_seq = tokenized_text[start_idx : start_idx + self.config.SEQ_LEN]
            target_seq = tokenized_text[start_idx + 1 : start_idx + self.config.SEQ_LEN + 1]

            step_loss = self.train_step(input_seq, target_seq)
            running_loss += step_loss

            if step % self.log_interval == 0:
                average_loss = running_loss / self.log_interval
                print(f"Step {step} | loss={average_loss:.4f}")
                running_loss = 0.0

            if step % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(
                    path=os.path.join(self.config.CHECKPOINT_DIR, "latest.pth"),
                    step=step
                )
    
    def save_checkpoint(self, path, step):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_step = checkpoint.get("step", 0)
        self.model.train()
        print(f"Model loaded from {path}")

