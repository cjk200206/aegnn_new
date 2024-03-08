import logging
import os
import pytorch_lightning as pl
import torch
import wandb


class FullModelCheckpoint(pl.callbacks.ModelCheckpoint):
    FILE_EXTENSION = ".pt"

    def _save_model(self, trainer: pl.Trainer, filepath: str) -> None:
        trainer.dev_debugger.track_checkpointing_history(filepath)
        if trainer.should_rank_save_checkpoint:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)
        # torch.save(trainer.model.state_dict(), filepath)
        trainer.save_checkpoint(filepath) #重新修改save,用pytorch_lighting
        logging.debug(f"Save model checkpoint @ {filepath}")
