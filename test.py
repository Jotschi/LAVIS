from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

image = Image.open("test.jpg").convert("RGB")
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, processor, _ = load_model_and_preprocess(
    name="Blip2TinyStories", model_type="pretrain_tinystories33m", is_eval=True, device=device
)


inputs = processor(image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

