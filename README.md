---
license: gpl-3.0
---

See [Hugging Face repo ](https://huggingface.co/Yairama/alpaca-miner)

[Mining Dataset](https://huggingface.co/datasets/Yairama/alpaca_miner_dataset)

## Trying to make a Fine tuning of alpaca using a mining dataset.

I will continue making the fine tuning when get a better GPU jaja.

### How to use:


Get the base model and the fine tuned model

```python
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

base_model = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "Yairama/alpaca-miner")
```


Making Predictions
```python
# Generate responses
def generate(instruction, input=None):
    if input:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(temperature=0.2, top_p=0.75, num_beams=4),
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=216
    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        print(output.split("### Response:")[1].strip())

generate("What is reclamation in mining")

#Reclamation in mining refers to the process of restoring land that has been disturbed by mining activities to its original state or better. It involves restoring the topography, vegetation, and ecosystems that were affected by the mining operations, and returning the land to its pre-mining condition or an even better state. ⁇ 

#Reclamation is an important part of the mining process, as it ensures that the land can be used for other purposes after the mining operations have been completed. It also helps to minimize the environmental impact of mining, by restoring the land to its natural state and preventing erosion and contamination of water sources. ⁇ 

#Reclamation techniques vary depending on the type of mining and the type of land that has been disturbed. For example, in open-pit mining, reclamation involves removing the overburden and restoring the land to its original contours. In underground mining, reclamation involves sealing the mine
```


Special thanks to:
[Spanish alpaca lora examples](https://huggingface.co/bertin-project/bertin-alpaca-lora-7b)
[Venelin Valkov tutorial](https://www.youtube.com/watch?v=4-Q50fmq7Uwc)

