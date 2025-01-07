import chainlit as cl
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your fine-tuned model and tokenizer
model_name = "krishkpatil/indian_legal_text_llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@cl.on_message
async def main(message: str):
    # Tokenize the input message
    inputs = tokenizer(message, return_tensors="pt")
    
    # Generate a response using the model
    outputs = model.generate(**inputs, max_length=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Send the response back to the user
    await cl.Message(content=response).send()
