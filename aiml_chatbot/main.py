from flask import Flask, render_template, request, jsonify
import aiml
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/ask", methods=['POST'])
def ask():
    message = request.form['messageText'].encode('utf-8').strip()

    kernel = aiml.Kernel()

    if os.path.isfile("bot_mind.brn"):
        kernel.bootstrap(brainFile="bot_mind.brn")
    else:
        kernel.bootstrap(learnFiles=os.path.abspath("aiml/std-startup.xml"), commands="load aiml b")
        kernel.saveBrain("bot_mind.brn")

    # kernel now ready for use
    while True:
        if message == "quit":
            exit()
        elif message == "save":
            kernel.saveBrain("bot_brain.brn")
        else:
            bot_response = kernel.respond(message)
            return jsonify({'status': 'OK', 'answer': bot_response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
