from flask import Flask, request, redirect, url_for, make_response
import os
import json
import subprocess
import string
import random
import glob

# import dilation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/ubuntu/uploads/'
app.config['RESULTS_FOLDER'] = '/home/ubuntu/results/'

@app.route("/")
def this_works_endpoint():
    return "THIS WORKS"

@app.route("/upload_video", methods = ['GET', 'POST'])
def upload_video_endpoint():
    print(request.form)
    print(request.files)
    file = request.files['file']
    stamps = request.form['stamps']

    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # for simple asynch processing because this is slow
    confirmation_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(32))
    subprocess.Popen(['python3', 'dilation.py', os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['RESULTS_FOLDER'], confirmation_number), stamps])

    return json.dumps({'confirmation_number': confirmation_number})

@app.route("/get_result", methods = ['GET', 'POST'])
def get_result_via_conf_number():
    confirmation_number = request.headers['confirmation_number']

    try:
        with open(os.path.join(app.config['RESULTS_FOLDER'], confirmation_number), 'r') as result_file:
            results = result_file.read()
            results = json.loads(results)
            results['finished'] = True
    except FileNotFoundError as e:
        return json.dumps({'finished': False})

    return json.dumps(results)

@app.route("/get_most_recent", methods = ['GET'])
def get_most_recent_result():
    list_of_files = glob.glob(app.config['RESULTS_FOLDER'] + '*')
    latest_file = max(list_of_files, key = os.path.getctime)

    with open(latest_file) as opened_latest_file:
        results = opened_latest_file.read()
        results = json.loads(results)
        breaking_point = results['breaking_point']

    return str(breaking_point)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
