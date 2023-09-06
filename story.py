import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_story():
    user_prompt = entry.get()
    generated_text = "Generated story:\n\n" + generate_story_text(user_prompt)
    result_label.config(text=generated_text)

def generate_story_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text based on the input prompt
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode and return the generated story
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Create the main application window
app = tk.Tk()
app.title("Storytelling Application")

# Create UI elements
label = tk.Label(app, text="Enter your story prompt:")
entry = tk.Entry(app, width=50)
generate_button = tk.Button(app, text="Generate Story", command=generate_story)
result_label = tk.Label(app, text="", wraplength=400)

# Place UI elements in the window
label.pack()
entry.pack()
generate_button.pack()
result_label.pack()

# Start the tkinter main loop
app.mainloop()
