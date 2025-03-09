from flask import Flask, render_template, jsonify
from flask_cors import CORS
import threading
import subprocess
import json, os

app = Flask(__name__)
CORS(app)

def run_gesture_program():
    subprocess.run(["python", "teste_cameras.py"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    threading.Thread(target=run_gesture_program).start()
    return jsonify({"status": "Programa iniciado!"})

@app.route('/get_recognition_result')
def get_result():
    if os.path.exists("result.json"):
        with open("result.json", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"raw_word": "-", "corrected_word": "-"})

if __name__ == '__main__':
    app.run(debug=True)
