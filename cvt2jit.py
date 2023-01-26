import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "orig"))

import torch
from models import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

cfg = "orig/cfg/yolov3_576x320.cfg"
weights = "model/jde_576x320_uncertainty.pt"
model = Darknet(cfg, nID=14455)
model.load_state_dict(torch.load(weights, map_location="cpu")["model"], strict=False)
model = model.to(device)
model.eval()

##################export###############
example_input = torch.rand(1, 3, 320, 576).to(device)
script_module = torch.jit.trace(model, example_input)
script_module.save("jit_convert/jde_576x320_torch14_gpu.pt")
##################end###############
