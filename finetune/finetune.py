#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic(
    "capture",
    "",
    "import os, re\nif \"COLAB_\" not in \"\".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)\n    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, \"0.0.34\")\n    !pip install sentencepiece protobuf \"datasets==4.3.0\" \"huggingface_hub>=0.34.0\" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n",
)


# In[ ]:


import os

import unsloth
from datasets import load_dataset
from google.colab import drive
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

drive.mount("/content/drive")


# In[23]:


from huggingface_hub import login

login(new_session=False)


# In[24]:


DATA_PATH = "/content/drive/MyDrive/belligpt/train_rhyme_suffix.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/belligpt/outputs/"

MODEL_NAME = "sapienzanlp/Minerva-3B-base-v1.0"


# In[25]:


max_seq_length = 512
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# In[ ]:


test_text = "pe’ mmojje una su’ pinitente"
tokens = tokenizer.tokenize(test_text)
print(tokens)


# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# In[ ]:


dataset = load_dataset("json", data_files=DATA_PATH, split="train")


# In[ ]:


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {
        "text": texts,
    }


dataset = dataset.map(formatting_prompts_func, batched=True)


# In[ ]:


import numpy as np

dataset_lengths = dataset.map(lambda x: {"len": len(tokenizer.encode(x["text"]))})
lens = dataset_lengths["len"]

max_found = max(lens)
median_len = int(np.median(lens))
mean_len = int(np.mean(lens))

print(f"Max: {max_found}")
print(f"Median: {median_len}")
print(f"Mean: {mean_len}")


# In[ ]:


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=14,
        num_train_epochs=3,
        save_steps=50,
        learning_rate=2e-4,
        seed=3407,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        output_dir=OUTPUT_DIR,
    ),
)


# In[ ]:


os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"
unsloth.USE_MODERN_PEFT = True
trainer.train()


# In[ ]:


model.save_pretrained(
    f"{OUTPUT_DIR}/rhyme_suffix/{MODEL_NAME.split('/')[-1]}_belli_adapter"
)
tokenizer.save_pretrained(
    f"{OUTPUT_DIR}/rhyme_suffix/{MODEL_NAME.split('/')[-1]}_belli_adapter"
)
