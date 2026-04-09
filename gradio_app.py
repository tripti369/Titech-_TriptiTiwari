import gradio as gr
from transformers import pipeline

# Load the model (use custom if available, else GPT-2)
model_path = "./models/final_model"
try:
    simplifier = pipeline("text-generation", model=model_path, device_map="auto")
except:
    simplifier = pipeline("text-generation", model="gpt2", device_map="auto")

def legal_simplifier(clause):
    prompt = f"Instruction: Simplify this legal clause.\nClause: {clause}\nSummary:"
    result = simplifier(prompt, max_new_tokens=50)
    return result[0]['generated_text'].split("Summary:")[-1].strip()

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📜 LegalEase: AI-Powered Legal Simplifier")
    gr.Markdown("Transform dense legal jargon into plain English.")

    with gr.Row():
        input_text = gr.Textbox(label="Paste Legal Clause Here", placeholder="e.g., 'Lessee shall indemnify Lessor...'")
        output_text = gr.Textbox(label="Plain English Summary")

    btn = gr.Button("Simplify", variant="primary")
    btn.click(fn=legal_simplifier, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch()