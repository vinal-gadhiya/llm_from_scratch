import os
import random

class Trainer:
    def __init__(self, model, config, optimizer, criterion):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.log_interval = 100

    def train_step(self, input_seq, target_seq):
        logits = self.model(input_seq)
        loss = self.criterion(
            logits.view(-1, self.config.VOCAB_SIZE),
            target_seq.view(-1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_iterations(self, tokenized_text, num_iterations):
        running_loss = 0.0
        for step in range(1, num_iterations+1):
            start_idx = random.randint(0, len(tokenized_text) - seq_len - 1)
            input_seq = torch.tensor(tokenized_text[start_idx : start_idx + seq_len])
            target_seq = torch.tensor(tokenized_text[start_idx + 1 : start_idx + seq_len + 1])
            step_loss = self.train_step(input_seq, target_seq)
            running_loss += step_loss

            if step % self.log_interval == 0:
                average_loss = running_loss / self.log_interval
                print(f"Step {step} | loss={average_loss:.4f}")
                running_loss = 0.0
    
    def save_checkpoint(self, path, step):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_step = checkpoint.get("step", 0)
        self.model.train()
        print(f"Model loaded from {path}")

