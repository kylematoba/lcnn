import os
import argparse
import contextlib

from typing import Type, List
import torch

def args_to_modelname(args: argparse.Namespace) -> str:
    model_filename = args.model_arch
    if args.regularizer is not None:
        model_filename += '_' + args.regularizer
    if args.dataset is not None:
        model_filename += '_' + args.dataset

    return model_filename


def get_named_modules_of_type(model: torch.nn.Module,
                               t: Type[torch.nn.Module]) -> List[torch.nn.Module]:
    return [__ for (_, __) in model.named_modules() if type(__) == t]


class ModelLogging():
    # Save trained and partially trained models in a directory
    def __init__(self,
                model_filename:str,
                save_path:str,
                logger,
                device) -> None:
        self.device = device
        self.logger = logger

        model_fullfilename = os.path.join(save_path, model_filename)
        self.model_fullfilename = model_fullfilename

        logger.info(f"Starting run -- checkpoints will be at {model_fullfilename}")


    def save_model_and_checkpoint(self, epoch, num_epochs, model):
        # Save the current model
        if epoch < num_epochs:
            epoch_fullfilename = f"{self.model_fullfilename}_{epoch}.pt"
        else:
            epoch_fullfilename = f"{self.model_fullfilename}.pt"

        self.logger.debug(f"Saving to {epoch_fullfilename}")
        torch.save(model.cpu().state_dict(), epoch_fullfilename)

        # delete previously stored model
        if epoch >= 1:
            with contextlib.suppress(FileNotFoundError):
                os.remove(f"{self.model_fullfilename}_{epoch - 1}.pt")
        model.to(self.device)