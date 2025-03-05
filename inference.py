from transformers import pipeline

# Load the quantized model on CPU
print("Loading quantized model...")
pipe = pipeline("text-generation", model="models/Llama-2-7b-chat-hf-bitnet", device="cpu")

user_prompt = input("Enter your prompt: ")

# Generate text based on user input
print("Generating text...")
output = pipe(
    user_prompt, 
    max_length=100,  
    temperature=0.5, 
)

# Print the generated text
print("\nGenerated Text:")
print(output[0]['generated_text'])