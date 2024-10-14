
# import transformers
# import torch
 
# pipeline = transformers.pipeline(
#     task="text-generation",
#     model="/home/data/wangzhefan/new/model/Meta-Llama-3-8B-Instruct", # "/root/models/Meta-Llama-3-8B",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# )
# import pdb
# pdb.set_trace()
# print(pipeline("Hey how are you doing today?"))



# import transformers
# import torch
# model_id = "./Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# )

# messages = [
#     {"role": "system", "content": "hello,You are a helpful human assistant!"},
#     {"role": "user", "content": "介绍一下中国,请用中文回答"},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )




# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM
# import transformers
# import torch
# import os

# model="/home/data/wangzhefan/new/model/Meta-Llama-3-8B-Instruct"
# # 定义模型名称
 
# tokenizer=AutoTokenizer.from_pretrained(model)
# # 使用预训练模型名称加载分词器
 
# llama=AutoModelForCausalLM.from_pretrained(model)
# # 使用预训练模型名称加载因果语言模型，并将其加载到指定的GPU设备上

# import time
 
# begin=time.time()
 
# input_text = "Write me a poem about maching learning."

# input_ids = tokenizer(input_text, return_tensors="pt").to(llama.device)

# import pdb
# pdb.set_trace()
# outputs = llama.generate(**input_ids)

# print(outputs)
 




# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# model_id = "/home/data/wangzhefan/new/model/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map='cuda',
# )

# messages = [
#     {"role": "system", "content": "You are an assistant who provides precise and direct answers."},
#     {"role": "user", "content": "In the sentence 'A boy is playing football', what is the exact action activity described? Provide only the exact phrase."},
# ]
# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
# import pdb
# pdb.set_trace()
# outputs = model.generate(
#     input_ids,
#     max_new_tokens=20,
#     eos_token_id=terminators,
#     do_sample=False,
#     temperature=0.0,
#     top_p=1.0,
# )
# response = outputs[0][input_ids.shape[-1]:]
# print(tokenizer.decode(response, skip_special_tokens=True)) # 输出 "playing football"





# from transformers.models.llama import LlamaConfig, LlamaModel
# import torch


# def run_llama():
#     llamaConfig = LlamaConfig(
#         vocab_size=32000,
#         hidden_size=4096 // 2,
#         intermediate_size=11008 // 2,
#         num_hidden_layers=32 // 2,
#         num_attention_heads=32 // 2,
#         max_position_embeddings=2048 // 2,
#     )
#     llamamodel = LlamaModel(config=llamaConfig)
#     # 构建输入
#     input_ids = torch.randint(low=0, high=llamaConfig.vocab_size, size=(4, 30))
#     res = llamamodel(input_ids)
#     print(res)

# if __name__ == "__main__":
#     run_llama()






# from transformers.models.llama import LlamaModel, LlamaConfig, LlamaTokenizerFast
# import torch
# from transformers import AutoTokenizer

  
# def run_llama(model_path):  
      
#     # 加载模型，并指定权重文件路径  
#     llamamodel = LlamaModel.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)    
      
#     # # 创建输入张量，这里我们随机生成一些输入ID  
#     # inputs_ids = torch.randint(low=0, high=llamaconfig.vocab_size, size=(4, 30))

#     # 你的文本数据  
#     texts = ["你好啊", "我很喜欢你"]

#     tokenizer.pad_token = tokenizer.eos_token  
#     # 使用分词器将文本转换为输入ID  
#     inputs_ids = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]  
   
      
#     # 运行模型
#     import pdb
#     pdb.set_trace()  
#     with torch.no_grad():  # 如果不需要梯度，可以使用torch.no_grad()来减少内存消耗  
#         outputs = llamamodel(inputs_ids)  
      
#     # 打印输出  
#     print(len(outputs))
#     print(len(outputs[0][0]))  
  
# if __name__ == '__main__':  
#     # 指定模型所在的目录路径  
#     model_directory = '/home/data/wangzhefan/new/model/Meta-Llama-3-8B-Instruct'  
#     run_llama(model_directory)




from transformers.models.llama import LlamaModel, LlamaConfig, LlamaTokenizerFast
import torch
from transformers import AutoTokenizer

  
def run_llama(model_path):  
      
    # 加载模型，并指定权重文件路径  
    llamamodel = LlamaModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)    
      
    # # 创建输入张量，这里我们随机生成一些输入ID  
    # inputs_ids = torch.randint(low=0, high=llamaconfig.vocab_size, size=(4, 30))

    # 你的文本数据  
    texts = ["你好啊", "我很喜欢你"]

    tokenizer.pad_token = tokenizer.eos_token  
    # 使用分词器将文本转换为输入ID  
    inputs_ids = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]  
   
      
    # 运行模型
    # import pdb
    # pdb.set_trace()  
    with torch.no_grad():  # 如果不需要梯度，可以使用torch.no_grad()来减少内存消耗  
        outputs = llamamodel(inputs_ids)  
    
      
    # 打印输出  
    # print(len(outputs))
    # print(len(outputs[0][0]))  
    # print(type(outputs))
  
if __name__ == '__main__':  
    # 指定模型所在的目录路径  
    model_directory = '/home/data/wangzhefan/new/model/Meta-Llama-3-8B-Instruct'  
    run_llama(model_directory)