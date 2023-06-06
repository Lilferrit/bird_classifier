from tqdm import tqdm
from typing import Dict, List, Tuple
from torch import nn
from torch.utils.data import DataLoader

import torch

def validate_model(
    vgg_model,
    val_loader,
    target_device: str = "cuda"
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    criterion.to(target_device)

    # One validation pass
    total_val_loss = 0
    total_val_num_correct = 0
    total_val_guesses = 0

    print(f"  validating...")

    with torch.no_grad():
      for batch_inputs, batch_outputs in tqdm(val_loader):
        # Move batch to device in use
        batch_inputs = batch_inputs.to(target_device)
        batch_outputs = batch_outputs.to(target_device)

        y_hat = vgg_model(batch_inputs)
        total_val_loss += criterion(y_hat, batch_outputs)

        labels = torch.argmax(y_hat, dim = 1)
        indicate_match = torch.where(
            labels == batch_outputs,
            1.0,
            0.0
        )
    
        count_correct = torch.sum(indicate_match).item()

        total_val_num_correct += count_correct
        total_val_guesses += len(indicate_match)

    current_val_loss = total_val_loss / len(val_loader)
    current_val_accuracy = total_val_num_correct / total_val_guesses

    print(f"  val loss: {current_val_loss}")
    print(f"  val accuracy: {current_val_accuracy}")

    return current_val_loss, current_val_accuracy

def train(
    vgg_model: nn.Module,
    optimizer: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_params_path: str,
    patience_interval: int = 10,
    num_epochs: int = 40,
    target_device: str = "cuda"
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()

    best_ce_loss = float("inf")
    curr_patience_interval = 0
    epoch_count = 0

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    vgg_model.to(target_device)
    criterion.to(target_device)

    for i in range(num_epochs):
        curr_patience_interval += 1
        epoch_count += 1

        # One training pass
        total_train_loss = 0
        total_num_correct = 0
        total_guesses = 0

        print(f"Epoch {epoch_count} of {num_epochs}:")
        print(f"  interval: {curr_patience_interval}")
        print(f"  training...")

        for batch_inputs, batch_outputs in tqdm(train_loader):
            optimizer.zero_grad()

            # Move batch to device in use
            batch_inputs = batch_inputs.to(target_device)
            batch_outputs = batch_outputs.to(target_device)

            # Compute y hat and loss
            y_hat = vgg_model(batch_inputs)
            loss = criterion(y_hat, batch_outputs)

            # Take training step
            loss.backward()
            optimizer.step()

            total_train_loss += loss

            # Compute train accuracy
            labels = torch.argmax(y_hat, dim = 1)
            indicate_match = torch.where(
                labels == batch_outputs,
                1.0,
                0.0
            )
            count_correct = torch.sum(indicate_match).item()

            total_num_correct += count_correct
            total_guesses += len(indicate_match)

        # One validation pass
        current_train_loss = total_train_loss / len(train_loader)
        current_train_accuracy = total_num_correct / total_guesses

        print(f"  train_loss: {current_train_loss}")
        print(f"  train accuracy: {current_train_accuracy}")

        current_val_loss, current_val_accuracy = validate_model(
            vgg_model,
            val_loader,
            target_device = target_device
        )

        print(f"  val loss: {current_val_loss}")
        print(f"  val accuracy: {current_val_accuracy}")
    
        train_losses.append(current_train_loss)
        train_accuracies.append(current_train_accuracy)

        val_losses.append(current_val_loss)
        val_accuracies.append(current_val_loss)

        if current_val_loss < best_ce_loss:
            curr_patience_interval = 0
            best_ce_loss = current_val_loss

            # Save model
            print(f"  new val loss record, saving new best model!")
            torch.save(vgg_model.state_dict(), model_params_path + "_best.pth")

        if curr_patience_interval >= patience_interval:
            break

        print(f"  saving current model!")
        torch.save(vgg_model.state_dict(), model_params_path + "_curr.pth")

        print("")

    return {
        'model': vgg_model,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
