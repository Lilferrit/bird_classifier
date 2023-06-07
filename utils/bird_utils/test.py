from tqdm import tqdm
from typing import Dict, List, Tuple
from torch import nn
from torch.utils.data import DataLoader

import torch

NAME_PATH = "/home/dubcar/bird_classifier/names/names.txt"

class NumToNum:
    def __init__(self, name_path: str = NAME_PATH):
        self.name_translate = dict()

        with open(NAME_PATH, "r") as f:
            for num, name in enumerate(f):
                self.name_translate[num] = name

    def get_name(self, num: int) -> str:
        return self.name_translate[num]
    
GLOBAL_NUMS = NumToNum()

def get_name(num: int) -> str:
    return GLOBAL_NUMS.get_name(num)

def get_predictions(
    vgg_model: nn.Module,
    test_loader: DataLoader,
    file_names: List[str],
    target_device: str = "cuda"
) -> Dict[str, List[float]]:
    predict_list = []

    print(f"  getting predictions...")

    with torch.no_grad():
        vgg_model.eval()

        for batch_inputs, batch_outputs in tqdm(test_loader):
            # Move batch to device in use
            batch_inputs = batch_inputs.to(target_device)

            y_hat = vgg_model(batch_inputs)
            labels = torch.argmax(y_hat, dim = 1)
        
            # Of course python has a + operator for list
            predict_list += labels.tolist()

        vgg_model.train()

    return {
        'predictions': predict_list,
        'file_names': file_names
    }

def predictions_to_csv(
    predict_dict: Dict[str, List],
    file_path: str
):
    with open(file_path, "a") as f:
        for file_name, prediction in zip(predict_dict['file_names'], predict_dict['predictions']):
            f.write(f"{file_name},{prediction}\n")