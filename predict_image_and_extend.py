from init_model import *

pipeline = StableDiffusionPipeline.from_pretrained("svjack/Stable-Diffusion-FineTuned-zh-v1")
pipeline.safety_checker = lambda images, clip_input: (images, False)
pipeline = pipeline.to("cuda")

path = "svjack/prompt-extend-chinese"
#### upload to huggingface
mt5_b_cpu = T5_B(path,
    device = "cpu")

def predict_one_with_prompt_extend(prompt):
    assert type(prompt) == type("")
    tail = mt5_b_cpu.predict_batch(
        [
            prompt
        ]
    )[0]
    assert type(tail) == type("")
    #prompt = '女孩们打开了另一世界的大门'
    prompt_extend = "{},{}".format(prompt, tail)
    image = pipeline(prompt, guidance_scale=7.5).images[0]
    return image


if __name__ == "__main__":
    predict_one_with_prompt_extend("女孩们打开了另一世界的大门")
