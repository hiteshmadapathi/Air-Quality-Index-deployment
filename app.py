
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Randomforest_model.pkl.', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='PM2.5 AQI Value should be {}'.format(output))

    df=pd.read_csv('xtest.csv')
    my_prediction=model.predict(df.values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)


if __name__ == "__main__":
    app.run(debug=True)