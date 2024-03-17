from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

#from lavis.models import model_zoo
#print(model_zoo)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_mistral", 
    model_type="caption_coco_mistral7b", 
    is_eval=True,
    device=device
)

directory = os.fsencode("test-images")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        raw_image = Image.open("test-images/" + filename).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        print("\n" + filename)
        print(model.generate({"image": image}))

#print("--------")
#inputs = vis_processors["eval"](raw_image, return_tensors="pt").to(device, torch.float16)
#generated_ids = model.generate(**inputs, max_new_tokens=20)
#generated_text = vis_processors["eval"].batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#print(generated_text)
