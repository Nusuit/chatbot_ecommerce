from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b")

def generate_response(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
