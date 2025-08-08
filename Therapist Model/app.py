from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
from therapist_model import generate_response
from transformers import pipeline

app = Flask(__name__)

# Emotion detection model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

chat_history = []

@app.route("/")
def home():
    return render_template("chat.html", chat_history=chat_history)

@app.route("/send", methods=["POST"])
def send():
    user_message = request.form["message"]

    # Detect emotion
    emotion_scores = emotion_analyzer(user_message)[0]
    top_emotion = max(emotion_scores, key=lambda x: x['score'])
    emotion_label = top_emotion['label']
    emotion_score = round(top_emotion['score'], 2)

    # Prepare dialog for model: just last few turns, labeled explicitly
    # Format: alternating user and therapist messages
    dialog_list = []
    # Keep last 4 turns (2 user + 2 bot) for context max
    history_to_use = chat_history[-4:] if len(chat_history) >= 4 else chat_history
    for entry in history_to_use:
        dialog_list.append(f"User: {entry['user']}")
        dialog_list.append(f"Therapist: {entry['bot']}")
    # Append current user message last
    dialog_list.append(f"User: {user_message}")

    bot_message = generate_response(dialog_list)

    chat_history.append({
        "user": user_message,
        "emotion": emotion_label,
        "score": emotion_score,
        "bot": bot_message,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    return redirect(url_for("home"))

@app.route("/report")
def report():
    emotion_count = {}
    for entry in chat_history:
        emotion_count[entry["emotion"]] = emotion_count.get(entry["emotion"], 0) + 1
    return render_template("report.html", emotion_count=emotion_count)

@app.route("/reset")
def reset():
    chat_history.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)

response = generate_response(["I feel overwhelmed and can't focus on anything. What can I do to calm my mind?"])
print(response)
