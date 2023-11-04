from flask import Flask,render_template,request
from pickle import load
import numpy as np
import pandas as pd
from keras.models import load_model
import sklearn
import statsmodels.api as smf
import os


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/single_pred', methods=['post'])
def single_ui():
    global loaded_model, prediction, name
    u_d = float(request.form.get('ud'))
    u_q = float(request.form.get('uq'))
    i_d = float(request.form.get('id'))
    pm = float(request.form.get('pm'))
    regressor = request.form.get('model')
    #print(regressor)
    #loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
    if regressor == 'ada':
        loaded_model = load(open('AdaBoost_model.sav', 'rb'))
        name = 'AdaBoost Regressor'
    elif regressor == 'random':
        loaded_model = load(open('Random_Forest.sav', 'rb'))
        name = 'Random Forest Regressor'
    elif regressor == 'decision':
        loaded_model = load(open('Decision_Tree.sav', 'rb'))
        name = 'Decision Tree Regressor'
    elif regressor == 'bag':
        loaded_model = load(open('Bagging.sav', 'rb'))
        name = 'Bagging Regressor'
    elif regressor == 'grad':
        loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
        name = 'Gradient Boost Regressor'
    elif regressor == 'stack':
        loaded_model = load(open('Stacking.sav', 'rb'))
        name = 'Stacking Regressor'
    elif regressor == 'knn':
        loaded_model = load(open('KNN.sav', 'rb'))
        name = 'KNN Regressor'
    elif regressor == 'cubic':
        loaded_model = load(open('Cubic.sav', 'rb'))
        name = 'Cubic Polynomial Regressor'
    elif regressor == 'nn':
        loaded_model = load_model('ANN.h5')
        name = 'Neural Network Regressor'
    data = {'u_d': u_d,
            'u_q': u_q,
            'i_d': i_d,
            'pm': pm}
    df = pd.DataFrame(data,index = [0])
    #prediction = (loaded_model.predict(df)[0]).round(6)
    if regressor in ['ada', 'decision Tree', 'random Forest', 'bag', 'knn', 'stack', 'grad',
                     'nn']:
        prediction = (loaded_model.predict(df)[0]).round(6)
    elif regressor == 'cubic':
        df['i_d_squared'] = df['i_d'] * df['i_d']
        df['u_q_squared'] = df['u_q'] * df['u_q']
        df['i_d_cube'] = df['i_d'] * df['i_d'] * df['i_d']
        df['u_q_cube'] = df['u_q'] * df['u_q'] * df['u_q']
        prediction = (loaded_model.predict(df)[0]).round(6)
    data['prediction'] = prediction
    data['name'] = name
    return render_template('single.html', data=data)

@app.route('/multi')
def multi():
    return render_template('multi.html')

@app.route('/multi_pred', methods=['post'])
def multi_ui():
    global loaded_model, prediction, name

    dataset = pd.read_csv(request.form.get('doc'))
    data = dataset[['u_d', 'u_q', 'i_d', 'pm']]

    regressor = request.form.get('model')
    # print(regressor)
    # loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
    if regressor == 'ada':
        loaded_model = load(open('AdaBoost_model.sav', 'rb'))
        name = 'AdaBoost Regressor'
    elif regressor == 'random':
        loaded_model = load(open('Random_Forest.sav', 'rb'))
        name = 'Random Forest Regressor'
    elif regressor == 'decision':
        loaded_model = load(open('Decision_Tree.sav', 'rb'))
        name = 'Decision Tree Regressor'
    elif regressor == 'bag':
        loaded_model = load(open('Bagging.sav', 'rb'))
        name = 'Bagging Regressor'
    elif regressor == 'grad':
        loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
        name = 'Gradient Boost Regressor'
    elif regressor == 'stack':
        loaded_model = load(open('Stacking.sav', 'rb'))
        name = 'Stacking Regressor'
    elif regressor == 'knn':
        loaded_model = load(open('KNN.sav', 'rb'))
        name = 'KNN Regressor'
    elif regressor == 'cubic':
        loaded_model = load(open('Cubic.sav', 'rb'))
        name = 'Cubic Polynomial Regressor'
    elif regressor == 'nn':
        loaded_model = load_model('ANN.h5')
        name = 'Neural Network Regressor'

    if regressor in ['ada', 'decision Tree', 'random Forest', 'bag', 'knn', 'stack', 'grad',
                     'nn']:
        prediction = loaded_model.predict(data)

    elif regressor == 'cubic':
        data['i_d_squared'] = data['i_d'] * data['i_d']
        data['u_q_squared'] = data['u_q'] * data['u_q']
        data['i_d_cube'] = data['i_d'] * data['i_d'] * data['i_d']
        data['u_q_cube'] = data['u_q'] * data['u_q'] * data['u_q']
        prediction = loaded_model.predict(data)

    data['Motor_speed'] = prediction

    data.to_csv(f'templates/output/Prediction_{name}.csv')
    result = data.to_csv().encode('utf-8')
    # filename = 'Predictions.csv'
    # save_location = os.path.join('templates', filename)
    # data.save(save_location)
    return render_template('multi.html', data=result)


if __name__ == '__main__':
    app.run(debug=True)