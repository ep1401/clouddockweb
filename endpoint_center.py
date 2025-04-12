import argparse
from typing import List, Dict
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---- Argument parsing ----
parser = argparse.ArgumentParser(description="Local model endpoint for Bittensor SN1 miner.")
parser.add_argument("--model", required=True, help="HuggingFace model name (e.g., EleutherAI/pythia-70m)")
parser.add_argument("--auth_token", default="changeme", help="Token required for miner auth")
parser.add_argument("--no_cuda", action="store_true", help="Force CPU inference")
args = parser.parse_args()

# ---- Load model ----
device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
print(f"Loading model: {args.model} on {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
model.eval()
print("✅ Model loaded successfully.")

# ---- Flask app ----
app = Flask(__name__)

@app.route("/", methods=["POST"])
def chat():
    request_data = request.get_json()

    # Authenticate
    if request_data.get("verify_token") != args.auth_token:
        return jsonify({"error": "Invalid token"}), 401

    # Format input
    messages: List[Dict[str, str]] = request_data.get("messages", [])
    if not messages:
        return jsonify({"response": "No input provided."})

    prompt = ""
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += "assistant:"

    # Run inference
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response_text = output_text.split("assistant:")[-1].strip()

    return jsonify({"response": response_text})

# ---- Run server ----
if __name__ == "__main__":
    print("🚀 Endpoint center running on http://0.0.0.0:8008")
    print(f"🔐 Token: {args.auth_token}")
    app.run(host="0.0.0.0", port=8008)
