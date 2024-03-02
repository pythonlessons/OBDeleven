# https://github.com/Arraxx/new-chatbot/blob/master/app/templates/index.html
from flask import Flask, render_template, request
from rag_model import RagChain
import yaml

app = Flask(__name__)

with open('configs.yaml', 'r') as file:
    configs = yaml.safe_load(file)

rag_chain = RagChain(**configs)

# Define a route to render the chat interface
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the bot responses
@app.route('/get')
def get_bot_response():
    # Get the message from the frontend
    user_message = request.args.get('msg')

    # Process the message and get the bot's response
    bot_response = process_message(user_message)  # You need to implement this function

    # Return the bot's response
    return bot_response

# Implement this function to process the user's message and generate the bot's response
def process_message(user_message):
    # Here, you can integrate your RAG solution or any other method to generate the bot's response
    return rag_chain.invoke(user_message)

if __name__ == '__main__':
    app.run(debug=True)