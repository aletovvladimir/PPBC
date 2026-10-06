import torch

import fastervit
from peft import LoraConfig, get_peft_model

import  argparse
torch.serialization.add_safe_globals([argparse.Namespace])

def LoraVIT(num_classes, r):
    model = fastervit.create_model(
        "faster_vit_0_224",
        pretrained=True,
        model_path="/tmp/faster_vit_0.pth.tar",
    )

    LoraConf = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
    )

    peft_model = get_peft_model(model, LoraConf)
    peft_model.head = torch.nn.Linear(peft_model.head.weight.shape[-1], num_classes)

    return peft_model
