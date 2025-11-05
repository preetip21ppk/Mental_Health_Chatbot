# 💬 AI/ML Mental Health Chatbot (SAAC)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/DeepLearning-TensorFlow-orange)
![Keras](https://img.shields.io/badge/Library-Keras-red)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)
![NLP](https://img.shields.io/badge/Model-NLP-green)
![SpeechRecognition](https://img.shields.io/badge/Voice-SpeechRecognition-yellow)

---

## 🧠 Project Overview

This project presents **SAAC (Smart AI-Assisted Counselor)** — an **AI-powered mental health chatbot** that provides 24/7 emotional support through **Natural Language Processing (NLP)** and **Deep Learning**.  
Built with **TensorFlow** and **Keras**, the chatbot classifies user intent and generates empathetic, context-aware responses to promote mental wellness and self-awareness.

The chatbot features a **Flask-based web interface**, supporting both **text and voice inputs** via the **Web Speech API** and **SpeechRecognition** library — ensuring accessibility for all users.

---

## ⚙️ Key Features

- 🧩 **NLP-driven Understanding:** Classifies user intent and context using deep learning.  
- 🗣️ **Voice & Text Support:** Enables real-time interaction through speech and text input.  
- 🎯 **High Accuracy:** Achieved **92% accuracy** in intent classification.  
- 💬 **Dynamic Conversations:** Context-aware responses based on trained intent models.  
- 🕒 **24/7 Availability:** Always-on emotional support system.  
- 🧾 **Chat History Tracking:** Logs user conversations for improved engagement and continuity.  

---

## 🧩 Architecture Overview

1. **Data Preprocessing**
   - Tokenized and lemmatized text using *NLTK*.
   - Constructed Bag-of-Words representations for input patterns.

2. **Model Training**
   - Built an intent classification model using **Keras Sequential API**.
   - Trained on labeled intents from `intents.json`.
   - Fine-tuned conversational quality using **BlenderBot 400M** (`finetune.py`).

3. **Deployment**
   - Integrated model into a **Flask API** (`app.py`).
   - Enabled **voice input processing** with SpeechRecognition.
   - Hosted a clean **web interface** for real-time chat.

---

## 🧠 Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP & Text Processing | NLTK, WordNetLemmatizer |
| Speech & Audio | SpeechRecognition, Web Speech API |
| Backend Framework | Flask |
| Model Fine-tuning | Hugging Face Transformers (BlenderBot) |
| Dataset | Custom intents + Counseling conversation dataset |

---

## 📈 Results

- **Intent Classification Accuracy:** 92%  
- **Response Relevance:** High contextual match with user queries  
- **Accessibility:** Fully supports both voice and text interactions  
- **Ethical Safeguards:** Integrated safety responses for sensitive topics  

---

## 💬 Example Conversation
- **User:** I feel really anxious today.
- **Bot:** I'm sorry to hear that. Taking a few deep breaths might help. Would you like to talk more about what’s making you anxious?

- **User:** I want to end my life.
- **Bot:** I'm really sorry you're feeling this way. You’re not alone. Please reach out for immediate help — call 988 (Suicide and Crisis Lifeline).


---

## 📂 Repository Structure

├── app.py # Flask web app for chatbot
├── forreal.py # Command-line chatbot interface
├── finetune.py # Fine-tuning script using BlenderBot
├── intents.json # Predefined intents and responses
├── chat_history.txt # Saved conversation logs
├── chatbot_model.keras # Trained neural network model
├── words.pkl / classes.pkl # Tokenized vocabulary and intent labels

## ❤️ Ethical Use Policy

 - This chatbot is not a replacement for professional therapy.
 - It is designed for emotional support and awareness only.
 - For emergencies, users are redirected to verified helpline numbers (e.g., 988).

## ⭐ If you found this project helpful, please star ⭐ the repository and connect with me on LinkedIn!
