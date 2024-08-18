import argparse
import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image as image
from sconf import Config
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer

import pipeline.interface as interface
from mplug_owl.processing_mplug_owl import MplugOwlProcessor
from pipeline.data_utils.processors.builder import build_processors
from pipeline.utils import add_config_args, set_args


def argconfig():
    # copied from ureader evaluation code

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default=None,
        help="Path to the trained checkpoint. If given, evaluate the given weights instead of the one in hf model.",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="./checkpoints/ureader",
        help="Path to the huggingface model",
    )
    args = parser.parse_args()
    config = Config("configs/sft/release.yaml")
    add_config_args(config, args)
    set_args(args)


def testloop(
    testnr: str,
    pretrained: str = "Mizukiluke/ureader-v1",
    imgloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/test",
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../results",
):
    argconfig()
    # model, tokenizer, processor = interface.get_model(pretrained)
    if torch.cuda.is_available():
        # model.half()
        device = torch.device("cuda")
        model, tokenizer, processor = interface.get_model(pretrained, use_bf16=True)
        config = Config("configs/sft/release.yaml")
        tokenizer1 = LlamaTokenizer.from_pretrained(pretrained)
        image_processor = build_processors(config["valid_processors"])["sft"]
        processor1 = MplugOwlProcessor(image_processor, tokenizer1)
        model.half()
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    imgs = glob.glob(f"{imgloc}/*.jpg")
    # text = "Transcribe the texts in the image."
    text = "Recognize all the texts in the image."
    prompt = f"Human: <image>\nHuman: {text}\nAI: "
    # prompts = [prompt for i in range(len(imgs))]
    # print(type([prompt]))
    dataset = imgloc.split("/")[-1]
    for impath in tqdm(imgs):
        # print(type(prompts), type(imgs))
        target = impath.split("/")[-1].split(".")[-2]
        sentences = interface.do_generate(
            prompt, [impath], model=model, tokenizer=tokenizer1, processor=processor1
        )
        # print(sentences)
        saveloc = f"{targetloc}/{pretrained}/{dataset}/{testnr}"
        savefile = f"{targetloc}/{pretrained}/{dataset}/{testnr}/{target}.json"
        os.makedirs(saveloc, exist_ok=True)
        with open(savefile, "w") as out:
            out.write(json.dumps(sentences, indent=4))


if __name__ == "__main__":
    # img = image.open(f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg").convert("RGB")
    # print(img.size)
    testloop("llama_tokenizer 2_different prompt_")
    # device = torch.device("cuda")
    # print(torch.cuda.is_available(), torch.cuda.current_device(), device)
    # print(torch.special.erfinv(torch.tensor([0, 0.5, -1.]).to(device, torch.float16)))
