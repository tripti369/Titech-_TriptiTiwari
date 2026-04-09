from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()

# Load your fine-tuned model
# Note: Use "gpt2" if you haven't finished training yet
model_path = "./models/final_model" 
try:
    simplifier = pipeline("text-generation", model=model_path, device_map="auto")
except:
    print("Custom model not found, using base model.")
    simplifier = pipeline("text-generation", model="gpt2", device_map="auto")

@app.post("/simplify")
async def simplify(clause: str):
    prompt = f"Instruction: Simplify this legal clause.\nClause: {clause}\nSummary:"
    result = simplifier(prompt, max_new_tokens=50)
    return {"summary": result[0]['generated_text'].split("Summary:")[-1].strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)