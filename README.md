---
license: gpl-3.0
---


## Trying to make a Fine tuning of alpaca using a mining dataset.

I will continue making the fine tuning when get a better GPU jaja.

### How to use:


Get the base model and the fine tuned model

```
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

base_model = "decapoda-research/llama-7b-hf"
tokenizer = LLaMATokenizer.from_pretrained(base_model)
model = LLaMAForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "Yairama/alpaca-miner")
```


Making Predictions
```
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
        max_new_tokens=256
    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        print(output.split("### Response:")[1].strip())

generate("Escribe un correo electrónico dando la bienvenida a un nuevo empleado llamado Manolo.")
# Estimado Manolo,
#
# ¡Bienvenido a nuestro equipo! Estamos muy contentos de que hayas decidido unirse a nosotros y estamos ansiosos por comenzar a trabajar juntos. 
#
# Nos gustaría darte las gracias por tu interés en nuestro equipo y esperamos que tengas un gran tiempo aquí. 
#
# Si tienes alguna pregunta o duda, no dudes en contactarnos. 
#
# Atentamente, 
# Equipo de [Nombre del Departamento]

```


Special thanks to:
![Spanish alpaca lora examples](https://huggingface.co/bertin-project/bertin-alpaca-lora-7b)
![Venelin Valkov tutorial](https://www.youtube.com/watch?v=4-Q50fmq7Uwc)

