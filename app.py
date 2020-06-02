import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    feature_name = ['total_bill','tip','size','sex_Female','day_Fri','day_Sat',
                    'day_Sun','time_Dinner']
    Df = pd.DataFrame(final_features,columns = feature_name)
    prediction = model.predict(Df)
    
    output = prediction
    
    if output ==  1:
        status = 'Non-smoker'
    else:
        status = 'smoker'


    return render_template('index.html', prediction_text='Customer is:   {}'.format(status))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
#    '''
#    For direct API calls through request
#    '''
#    data = request.get_json(force =True)
#    prediction = model.predict([np.array(list(data.values()))])
#
#    output = round(prediction[0],2)
#
#    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    