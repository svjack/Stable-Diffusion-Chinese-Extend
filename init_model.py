from diffusers import StableDiffusionPipeline
#pipe = StableDiffusionPipeline.from_pretrained("svjack/Stable-Diffusion-FineTuned-zh-v0")
#pipe = StableDiffusionPipeline.from_pretrained("svjack/Stable-Diffusion-FineTuned-zh-v1")
#pipe = StableDiffusionPipeline.from_pretrained("svjack/Stable-Diffusion-FineTuned-zh-v2")
from transformers import T5Tokenizer, MT5ForConditionalGeneration

class T5_B(object):
    def __init__(self, model: str = "google/t5-large-ssm", device = 'cuda:0'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        if device == 'multigpu':
            self.model = MT5ForConditionalGeneration.from_pretrained(model).eval()
            self.model.parallelize()
        else:
            self.model = MT5ForConditionalGeneration.from_pretrained(model).to(device).eval()

    def predict(self, question: str):
        device = 'cuda:0' if self.device == 'multigpu' else self.device
        encode = self.tokenizer(question, return_tensors='pt').to(device)
        answer = self.model.generate(encode.input_ids)[0]
        decoded = self.tokenizer.decode(answer, skip_special_tokens=True)
        return decoded

    def predict_batch(self, question_list):
        assert type(question_list) == type([])
        device = 'cuda:0' if self.device == 'multigpu' else self.device
        encode = self.tokenizer(question_list, return_tensors='pt', padding = True).to(device)
        answer = self.model.generate(**encode)
        #return answer
        decoded = [self.tokenizer.decode(ans, skip_special_tokens=True) for ans in answer]
        #decoded = self.tokenizer.decode(answer, skip_special_tokens=True)
        return decoded
