from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(r"C:\Projects\Doctor-Bot\bot"), 'bot'))
from bot.retrieval_gen import generation
from bot.ingest_data import ingest


app = Flask(__name__)

load_dotenv()

vector_store=ingest("done")
chain=generation(vector_store)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result=chain.invoke(input)
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    print("Current Working Directory: ", os.getcwd())

    app.run(debug= True)