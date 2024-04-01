from flask import Flask,render_template, request
import pickle as pk
import numpy as np
import pandas as pd
import bz2file as bz2

app = Flask(__name__)

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pk.load(data)
    return data

model = decompress_pickle('benstokes.pbz2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    example = {
        'BattingTeam': [request.form['BattingTeam']],
        'BowlingTeam': [request.form['BowlingTeam']],
        'City': [request.form['City']],
        'runs_left': [float(request.form['runs_left'])],
        'balls_left': [float(request.form['balls_left'])],
        'wickets_left': [float(request.form['wickets_left'])],
        'current_run_rate': [float(request.form['current_run_rate'])],
        'required_run_rate': [float(request.form['required_run_rate'])],
        'target': [float(request.form['target'])]
    }

    input_df = pd.DataFrame(example)

    #input_data = np.array([[batting_team, bowling_team, city,  runs_left, balls_left, wickets_left, current_run_rate, required_run_rate, target]])

    predicted_result = model.predict(input_df)
    ans = ''
    if predicted_result[0] == 1 :
        ans = example['BattingTeam']
    else:
        ans = example['BowlingTeam']

    return render_template('result.html', prediction=ans[0])

if __name__ == '__main__':
    app.run(debug=True)