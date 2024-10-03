import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("baofamily/gemma-2b-finetuned-model-llama-factory")
    model = AutoModelForCausalLM.from_pretrained("baofamily/gemma-2b-finetuned-model-llama-factory")
    return tokenizer, model

tokenizer, model = load_model()

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    output = model.generate(inputs['input_ids'], max_length=500)
    return tokenizer.decode(output[0], skip_special_tokens=True)


st.title("Chefridge Recipe Generator")

input_ingredients = st.text_input("In your Refrigerator: ")
instruction = "Chefridge has spotted these ingredients in your fridge! What delicious recipe can you make with:"

if st.button("Generate Recipe"):
    if input_ingredients:
        prompt = f"{instruction}\n{input_ingredients}"
        
        with st.spinner("Generating recipe..."):
            try:
                generated_text = generate_text(prompt)
                st.write("Done! ")
                st.write(generated_text) 
            except Exception as e:
                st.error(f"An error occurred: {e}") 
    else:
        st.warning("Please enter some ingredients!")
