from init_model import *

pipeline = StableDiffusionPipeline.from_pretrained("svjack/Stable-Diffusion-FineTuned-zh-v1")
pipeline.safety_checker = lambda images, clip_input: (images, False)
pipeline = pipeline.to("cuda")

if __name__ == "__main__":
    prompt = '女孩们打开了另一世界的大门'
    image = pipeline(prompt, guidance_scale=7.5).images[0]
    image

    x = "女孩们打开了另一世界的大门:,动漫、复杂、敏锐焦点、插图、高度详细、数字绘画、概念艺术、配制、由,和,制作"
    image = pipeline(x, guidance_scale=7.5).images[0]
    image
