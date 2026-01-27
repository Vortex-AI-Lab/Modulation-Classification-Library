from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, List, Dict

import time
import sys
import os
from os import path

from accelerate import Accelerator
from colorama import Fore, Style
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer

from sklearn.metrics import confusion_matrix

from model import MCformer

from utils.dataset import (
    RML2016DataLoader,
    RML2018DataLoader,
    PreTrainingDataLoader,
)
from utils.tools import (
    get_loss_fn,
    EarlyStopping,
    OptimInterface,
    logging_results,
    print_configs,
)


class BaseExperiment(ABC):
    """Base class for experiments."""

    def __init__(self, configs, accelerator: Accelerator, setting: str) -> None:
        super().__init__()
        self.configs = configs
        self.accelerator = accelerator

        self.dataset = configs.dataset
        self.snr = configs.snr

        # The mode of the experiment: "supervised" or "pre-training"
        self.mode = configs.mode

        # The hyper-parameters for data loading
        self.batch_size = configs.batch_size
        self.shuffle = configs.shuffle

        # The checkpoint directory
        self.checkpoint_dir = configs.checkpoint
        self.checkpoint_path = path.join(self.checkpoint_dir, setting)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Get the number of classes in the dataset
        self.n_classes = None
        self.class_list = None

        # The file path of the training dataset for supervised learning
        # Or the testing dataset for unsupervised learning (zero-shot learning)
        self.data_path = configs.data_path

        # The root path of the data for model fine-tuning or large scale pre-training
        self.root_path = configs.root_path

    @abstractmethod
    def run(self, setting: str) -> None:
        """Run the experiment."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def model_dict(self) -> Dict[str, nn.Module]:
        """Return a dictionary of available models."""
        return {
            "MCformer": MCformer.Model,
            # Add other models here as needed
        }

    def build_model(self, name: str = "SpectrumTime") -> nn.Module:
        """Build the model for training."""
        # Check if the model name is valid
        assert name in self.model_dict, f"Model {name} is not supported."

        self.accelerator.print(f"Building the model: {name}", end=" -> ")

        # Create the model for experiment
        model = self.model_dict[name].Model(self.configs)
        self.accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)

        return model

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any, Any]:
        """Load the dataset for training, validation, and testing."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _load_optimizer(self):
        """
        Load the optimizer for training.

        Creating an optimizer requires passing in the model's parameters.
        """
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."
        if self.optim is not None:
            return self.optim.load_optimizer(parameters=self.get_trainable_params())
        else:
            raise ValueError("Optimizer interface is not initialized.")

    def _load_scheduler(self, optimizer: Optimizer, loader_len: int):
        """
        Load the learning rate scheduler for training.
        """
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."

        if self.optim is not None:
            return self.optim.load_scheduler(optimizer, loader_len)
        else:
            raise ValueError("Optimizer interface is not initialized.")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Obtain trainable parameters of the model in experiment."""
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Obtain the number of trainable parameters"""
        return sum(p.numel() for p in self.get_trainable_params())

    def get_learning_rate(self, optimizer: Optimizer) -> float:
        """Get the current learning rate of the optimizer"""
        return optimizer.param_groups[0]["lr"]

    def _check_loss(self, loss: torch.FloatTensor) -> None:
        """Check the training and validation losses to avoid gradient explosion."""
        if not torch.isfinite(loss):
            self.accelerator.print(
                Fore.RED
                + "WARNING: non-finite loss, ending training!"
                + Style.RESET_ALL
            )
            sys.exit(1)

    def print_start_message(self, time_now: str) -> None:
        """Prints a message indicating the start of the experiment."""
        print_configs(
            accelerator=self.accelerator,
            time_now=time_now,
            config={  # The main configs parameters to be printed
                "seq_len": self.configs.seq_len,
                "pred_len": self.configs.pred_len,
                "epochs": self.configs.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.configs.learning_rate,
                "optimizer": self.configs.optimizer,
                "scheduler": self.configs.scheduler,
                "criterion": self.configs.criterion,
            },
            experiment_name="Spectrum Prediction",
            model_name=self.model_name,
            dataset=self.configs.dataset_name,
            mode=self.mode,
            print_separator=True,
        )

    def confusion_matrix(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """计算混淆矩阵"""
        # 展平并转为numpy数组
        predictions_np = predictions.flatten().cpu().numpy()
        targets_np = targets.flatten().cpu().numpy()

        # 计算混淆矩阵（labels确保类别顺序）
        labels = np.arange(self.n_classes) if self.n_classes else None
        cm_np = confusion_matrix(predictions_np, targets_np, labels=labels)

        # 转回PyTorch张量
        cm_tensor = torch.from_numpy(cm_np)

        return cm_tensor

    def save_results(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        accuracy: Union[float, torch.Tensor, np.ndarray],
        confusion_matrix: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Save experimental results to a specified path."""
        results_path = self.checkpoint_path + "/results.pth"

        # Wait for all processes to finish before saving
        self.accelerator.wait_for_everyone()

        # Only main process saves the results
        if self.accelerator.is_main_process:
            # Save the model inputs, predictions, targets, MSE, and MAE
            self.accelerator.save(
                obj={
                    "predictions": predictions,
                    "targets": targets,
                    "accuracy": accuracy,
                    "confusion_matrix": confusion_matrix,
                },
                f=results_path,
                safe_serialization=True,
            )
        self.accelerator.print("Test results saved to " + results_path)

    @property
    def logging_headers(self) -> List[str]:
        """The Log header for the results of the csv."""
        return [
            "Timestamp",
            "Dataset",
            "Model",
            "Mode",  # The mode for the model
            "Accuracy",  # The Accuracy of Classification
            "split_ratio",
            "seq_len",
            "Loss_function",
            "Optimizer",
            "Learning_rate",
            "Batch_size",
            "setting",
            "seed",
            "path",  # The path to the saved checkpoint and results
        ]


class SupervisedExperiment(BaseExperiment):

    def __init__(
        self, configs, accelerate: Accelerator, setting: str, time_now: str
    ) -> None:
        super().__init__(configs=configs, accelerator=accelerate, setting=setting)
        self.print_start_message(time_now=time_now)

        # Load the data for training and testing
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

        # Build the deep learning model
        self.model_name = configs.model
        self.model = self.build_model(name=self.model_name)

        # Load the optimizer and scheduler for model training
        self.optim = OptimInterface(configs=configs, accelerator=accelerator)
        self.optimizer = self._load_optimizer()
        self.scheduler = self._load_scheduler(
            optimizer=self.optimizer,
            loader_len=len(self.train_loader),
        )

        # Create the loss function for classification
        self.criterion = get_loss_fn(configs.criterion)

    @property
    def dataset_dict(self) -> Dict[str]:
        return {
            "RML2016a": RML2016DataLoader,
            "RML2016b": RML2016DataLoader,
            "RML2018a": RML2018DataLoader,
            "HisarMod2019.1": None,
        }

    def load_data(
        self,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:

        assert (
            self.dataset in self.dataset_dict.keys()
        ), f"{self.dataset} is not in {self.dataset_list}!"

        data_interface = self.dataset_dict[self.dataset](configs=self.configs)
        self.class_list = data_interface.class_list
        self.n_classes = len(self.class_list)
        self.configs.n_classes = self.n_classes

        train_loader, val_loader, test_loader = data_interface.get_data_loader()

        return train_loader, val_loader, test_loader

    def train(
        self, epoch: int
    ) -> Tuple[
        Union[float, torch.Tensor, np.ndarray], Union[float, torch.Tensor, np.ndarray]
    ]:
        """
        A unified approach for training self-supervised models one epoch.

        We will use a loop to repeatedly call this method to alternate between training and validating the model.
        """
        # Set the model to train mode
        self.model.train()
        num_samples = 0

        # Loop over the batches
        train_loss = torch.zeros(1, device=self.accelerator.device)
        train_accuracy = torch.zeros(1, device=self.accelerator.device)

        data_loader = tqdm(self.train_loader, file=sys.stdout)

        for step, (batch_x, batch_y) in enumerate(data_loader, 1):
            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            num_samples += batch_y.size(0)

            loss = self.criterion(outputs, batch_y)
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.scheduler.step()

            _, predicted = torch.max(outputs, dim=1)
            train_accuracy += torch.eq(predicted, batch_y).sum()

            train_loss += loss.item()

            data_loader.desc = f"[Train Epoch {epoch}] Loss: {round(train_loss.item() / num_samples, 4)}, Accuracy: {round(train_accuracy.item() / num_samples, 4)}"

        self._check_loss(loss=train_loss)

        return train_loss.item() / num_samples, train_accuracy.item() / num_samples

    def val(
        self, epoch: int
    ) -> Tuple[
        Union[float, torch.Tensor, np.ndarray], Union[float, torch.Tensor, np.ndarray]
    ]:
        """
        Validation methods for self-supervised learning-based spectrum prediction models in each epoch.

        We use this method to determine whether an early stop operation needs to be performed.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize validation loss
        val_loss = torch.zeros(1, device=self.accelerator.device)
        val_accuracy = torch.zeros(1, device=self.accelerator.device)

        num_samples = 0

        # Iterate over validation data
        data_loader = tqdm(self.val_loader, file=sys.stdout)

        with torch.no_grad():
            # Iterate over batches
            for step, (batch_x, batch_y) in enumerate(data_loader, 1):
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                val_accuracy += torch.eq(predicted, batch_y).sum()

                # accumulate the metrics for time series forecasting
                data_loader.desc = f"[Test  Epoch {epoch}] Loss: {round(val_loss.item() / num_samples, 4)}, Accuracy: {round(val_accuracy.item() / num_samples, 4)}"

        return val_loss.item() / num_samples, val_accuracy.item() / num_samples

    def test(self):
        """
        A unified interface for auto modulation classification testing.

        To ensure the reliability of the experiment,
        all base classes used for verifying results uniformly call this method to obtain the results.

        The created `data_loader` object in PyTorch needs to be passed in.
        """
        # loading the model parameters from checkpoint
        self.accelerator.print(
            "🚀 "
            + Fore.BLUE
            + "Loading the best model for testing..."
            + Style.RESET_ALL
        )
        self.accelerator.load_state(self.checkpoint_path)

        # Create the lists to store predictions and targets
        inputs = []
        predictions = []
        targets = []
        time_list = []

        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():

            for batch_x, batch_y in tqdm(self.test_loader):
                time_start = time.time()
                outputs = self.model(batch_x)
                time_end = time.time()

                time_list.append(time_end - time_start)

                inputs.append(batch_x)
                predictions.append(outputs)
                targets.append(batch_y)
