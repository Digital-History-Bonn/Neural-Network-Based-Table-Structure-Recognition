import torch
from  transformers import AutoModelForObjectDetection, DetrImageProcessor, TableTransformerForObjectDetection

def inference(modelpath:str=None):
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition" if not modelpath else modelpath)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return

