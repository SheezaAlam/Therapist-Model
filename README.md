## Therapist Chatbot

## Overview

This is a compassionate AI-powered therapist chatbot web app built with **Flask** and **Hugging Face's GODEL large seq2seq model**. The chatbot detects user emotions in real-time and responds with empathetic, supportive, and practical coping advice to help users manage emotional distress, anxiety, and racing thoughts.

---

## Features

* **Real-time Emotion Detection:**
  Uses Hugging Face's `j-hartmann/emotion-english-distilroberta-base` model to classify user text into emotions (e.g., sadness, fear, neutral, surprise) with confidence scores.

* **Empathetic Therapist Responses:**
  Powered by Microsoft’s `GODEL-v1_1-large-seq2seq` conversational model fine-tuned for goal-directed dialog, it generates thoughtful, compassionate therapist-style replies.

* **Custom Therapist Instruction:**
  The model is guided by a fixed system prompt instructing it to listen, validate emotions, and provide supportive coping guidance without offering medical advice.

* **Conversation History:**
  Maintains and displays the ongoing chat history with timestamps, user messages, detected emotions, and AI therapist replies.

* **Emotion Report:**
  Summarizes and counts detected emotions throughout the conversation to provide insight into the user’s emotional state over time.

* **Reset Chat:**
  Allows users to clear the conversation and start fresh at any time.

* **Local Model Hosting:**
  The GODEL model is downloaded and run locally with PyTorch to avoid API latency and limitations.

---

## Technologies Used

* Python 3.10+
* Flask (Web framework)
* Hugging Face Transformers (`transformers` library)
* PyTorch (for running the GODEL model)
* Requests (for API calls if needed)
* Jinja2 (Flask templating engine)
* HTML, CSS (frontend)
  
Project Structure
-----------------

TherapistModel/
│
├── app.py                 # Main Flask app with routes and logic
├── therapist_model.py     # Model loading and response generation logic
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── test_hf.py             # Test script to test Hugging Face model calls
│
├── templates/             # HTML templates for the Flask app
│   ├── chat.html          # Chat interface template
│   └── report.html        # Emotion report page
│
└── static/                # Static files (CSS, JS, images)
    └── chat.css           # Styling for chat UI


## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/TherapistModel.git
cd TherapistModel
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install required Python packages

```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` includes:

```
flask
transformers
torch
requests
```

### 4. Download and cache Hugging Face models

The app will automatically download models on first run. This might take some time (\~3GB for GODEL).
**Note:** Ensure you have enough disk space (\~5GB) and a stable internet connection.

Optional: Install `huggingface_hub[hf_xet]` for faster downloads on Windows

```bash
pip install huggingface_hub[hf_xet]
```

### 5. Set your Hugging Face API token (optional)

If you want to use the Hugging Face Inference API instead of local model loading, set your token:

```python
HF_TOKEN = "your_huggingface_token_here"
```

For local model usage, no token is required.

### 6. Run the Flask app

```bash
python app.py
```

By default, it runs on `http://127.0.0.1:5000/`.

Open this URL in your browser to start chatting with the therapist bot.

---

## Usage

* Type your messages into the chat input.
* The bot detects your emotional state and replies empathetically.
* View emotion summaries on the **Report** page.
* Reset the chat anytime with the **Reset** button.

---

## Notes and Limitations

* The therapist model provides emotional support but **is not a substitute for professional mental health care**.
* Running the GODEL model locally requires good CPU/GPU and RAM. It can be slow on CPU.
* Emotion detection and responses depend on model quality and may sometimes seem generic or repetitive.
* The app runs in Flask development mode and should be deployed with a production WSGI server for production use.

---

## Citation

If you use this project or the models in your research, please cite the following paper:

```bibtex
@misc{peng2022godel,
author = {Peng, Baolin and Galley, Michel and He, Pengcheng and Brockett, Chris and Liden, Lars and Nouri, Elnaz and Yu, Zhou and Dolan, Bill and Gao, Jianfeng},
title = {GODEL: Large-Scale Pre-training for Goal-Directed Dialog},
howpublished = {arXiv},
year = {2022},
month = {June},
url = {https://www.microsoft.com/en-us/research/publication/godel-large-scale-pre-training-for-goal-directed-dialog/},
}
```

---

## Contact

Email: [shizaalam50@gmail.com](mailto:shizaalam50@gmail.com)

#TherapistBot

#MentalHealthAI

#EmotionalAI

#ConversationalAI

#Chatbot

#AIForGood

#MentalHealthTech
#Python

#Flask

#HuggingFace

#OpenSource

#AIModel

#AIChatbot

#GPT

#AICommunity

#TechForGood

#AIResearch

#Innovation
