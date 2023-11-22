import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import random
import os
import config as c


from utils.dir import mkdir

os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.environ['CURL_CA_BUNDLE'] = ''

# "key words: primeval forest, beach", "garden", "suburb, tower"

# 初始化，定义
def init(device='cuda'):
    model_id = '/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a'
    # load/reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    return ldm_stable



def setup_seed(seed):
     # 设置随机数种子
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


pipe = init()
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device)


prompt_list = ['']
num_of_imgs_for_each_class = 2
img_save_dir = 'data/genearted_cover'


for prompt in prompt_list:
     img_dir = os.path.join(img_save_dir, prompt+'noprompt')
     if not os.path.exists(img_dir):
          os.makedirs(img_dir)


for prompt in prompt_list:
     for i in range(num_of_imgs_for_each_class):
          random_seed = i + 1    # the seed (10*i) is responsible for generating the i-th image
          setup_seed(random_seed)
          image = pipe(prompt).images[0]
          
          img_name = str(i) + '.png'
          img_dir = os.path.join(img_save_dir, prompt)
          image.save(os.path.join(img_dir, img_name))