from transformers import pipeline , AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
device='cuda' if torch.cuda.is_available() else 'cpu'

model_name='google/gemma-2b'

model=AutoModelForCausalLM.from_pretrained(model_name).to('cpu')

def infer(input_text="How to make a sandwich",no_of_tokens=500,device=device):
    tokeniser=AutoTokenizer.from_pretrained(model_name)
    
    tokeniser.chat_template = """{% for message in messages %}
        {% if message.role == "system" %}
            {{ message.content }}
        {% elif message.role == "user" %}
            {{ message.content }}
        {% elif message.role == "assistant" %}
            {{ message.content }}
        {% endif %}
    {% endfor %}
    """

    messages = [{"role": "user", "content": input_text}]
    prompt = tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokeniser(prompt, return_tensors="pt").to(device)
    print('generating...')
    output_ids = model.generate(input_ids.input_ids,max_length=no_of_tokens,do_sample=True,temperature=0.7,top_p=0.9)
    response = tokeniser.decode(output_ids[0], skip_special_tokens=True)
    return response