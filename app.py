from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data for training the sentiment analysis model
data = [
    ("Nice", "Nice"),
    ("sophorn", "sophorn"),
    ("I love how easy it is to use this product.", "positive"),
    ("I am disappointed with the results.", "negative"),
    ("Hi", "Hi"),
    ("Hello", "Hello"),
    ("Angry", "Angry"),
    ("How are you", "How are you"),
    ("How are you today", "How are you today"),
    ("Good bye","Good bye"),
    ("xnxx","xnxx"),
    ("fuck","fuck"),
    ("translate","translate"),
]

# Extracting texts and labels
texts, labels = zip(*data)

# Vectorizing the text data
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(texts)

# Training the Naive Bayes model
model = MultinomialNB()
model.fit(x, labels)

@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")  # Ensure 'chat.html' exists in the templates folder

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get the user input from the request
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "No message provided!"}), 400
        
        # Vectorize the user input and make a prediction
        user_vector = vectorizer.transform([user_input])
        sentiment = model.predict(user_vector)[0]
        
        # Generate response based on the sentiment prediction
        if sentiment == "Nice":
            response = "Thank you! We're glad to hear that!"
        elif sentiment == "sophorn":
            response = "Yes I know him, he is a gentalman and hunesman he alway help people around the world."
        elif sentiment == "Hi":
            response = "Hello! How can I assist you today?"
        elif sentiment == "Hello":
            response = "Hello! How may I help you today?"
        elif sentiment == "Angry":
            response = "Sorry to hear that you're angry. Please let me know how I can assist you better."
        elif sentiment == "How are you":
            response = "I'm good, thanks for asking! How can I help you today?"
        elif sentiment == "How are you today":
            response = "I'm doing great, thanks for asking! How can I assist you?"
        elif sentiment == "Good bye":
            response = "Yes good bye have a nice day and good luck your work. If you have something need I help you please tell me!"
        elif sentiment == "fuck":
            response = "Oh Dear!, Do you know this is bed word!. pleas tell what I can help you!"
        elif sentiment == "xnxx":
            response = "This is a website that have video sex,"
        elif sentiment =="translate":
            response = f'Here is a link: <a href="https://translate.google.com/" target="_blank">https://translate.google.com/</a>'

        else:
            response = "We're sorry, but I didn't understand that. Can you please rephrase?"

        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
