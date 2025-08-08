# therapist_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("therapist_model: device =", device)

# Load once at startup (may download model weights on first run)
print("therapist_model: Loading GODEL model/tokenizer (this can take several minutes)...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = model.to(device)
model.eval()
print("therapist_model: Model loaded.")

# Therapist-style system instruction (keeps the style consistent)
SYSTEM_INSTRUCTION = (
    "Do not include role labels like 'Therapist:' or 'User:' in your responses."
    "Instruction: You are a compassionate, professional therapist. "
    "Listen carefully, validate emotions, and offer supportive guidance. "
    "Always give at least 3 clear, practical coping strategies or exercises that the user can try immediately. "
    "Avoid repeating the same phrase more than once in a reply. "
    "Write in short, encouraging sentences. "
    "Do not give medical or legal advice; encourage seeking a professional when appropriate. "
)

def build_query_from_dialog(dialog_list, knowledge=""):
    """
    dialog_list: list of strings (alternating user and bot turns or just user turns)
    We'll join them using ' EOS ' as the GODEL example shows.
    """
    if not isinstance(dialog_list, list):
        dialog_list = [dialog_list]

    dialog_str = " EOS ".join(dialog_list)
    knowledge_part = f"[KNOWLEDGE] {knowledge}" if knowledge else ""
    query = f"{SYSTEM_INSTRUCTION} [CONTEXT] {dialog_str} {knowledge_part}"
    return query

def generate_response(dialog_list, knowledge="", max_length=200, min_length=30):
    """
    Generate therapist-style response given dialog_list (the conversation so far).
    Returns a text reply.
    """
    query = build_query_from_dialog(dialog_list, knowledge=knowledge)

    # Detect help-seeking intent and push for practical tips
    help_keywords = ["how", "what can i do", "any tips", "suggest", "help me", "manage", "cope"]
    last_utterance = dialog_list[-1].lower() if dialog_list else ""
    if any(kw in last_utterance for kw in help_keywords):
        query += " Please respond with 3-5 specific, actionable coping strategies, each in a separate bullet point."

    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.replace("Therapist:", "").replace("User:", "").strip()
    return reply
