from surrogates import get_llava_predict_model, GPT4PredictModel
from tester import test_jailbreak_attack_using_gpt4

llava = get_llava_predict_model(prompt="<image>\nUSER: What's the content of the image?\nASSISTANT:")
gpt4 = GPT4PredictModel()
test_jailbreak_attack_using_gpt4("./pgd_jailbreak/", [llava], gpt4)
