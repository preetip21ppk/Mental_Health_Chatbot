from flask import Flask, render_template, request, jsonify
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import random
import speech_recognition as sr

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

# Load trained model
model = tf.keras.models.load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent' : classes[r[0]], 'probability' : str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if not ints:
        return "Sorry, I didn't understand."
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    # Convert tag to lowercase for case-insensitive comparison
    tag_lower = tag.lower()
    
    for intent in list_of_intents:
        # Check if the intent tag matches the predicted tag (case-insensitive)
        if intent['tag'].lower() == tag_lower:
            if 'responses' in intent:
                result = random.choice(intent['responses'])
                break
            else:
                result = "Sorry, I'm not sure how to respond to that."
                break
    else:
        result = "Sorry, I'm not sure how to respond to that."

    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if 'message' in request.json:
        message = request.json['message']
    elif 'voice' in request.files:
        audio_file = request.files['voice']
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        try:
            message = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            message = "Sorry, I could not understand the audio."
        except sr.RequestError:
            message = "Sorry, there was an issue with the speech recognition service."
    else:
        return jsonify({'response': "No message or voice file detected."})

    ints = predict_class(message)
    response = getResponse(ints, intents)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
