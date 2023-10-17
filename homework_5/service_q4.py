import pickle as pkl
from flask import Flask, request, jsonify

model_file = './model1.bin'
dv_file = './dv.bin'

# read the file
with open(dv_file,'rb') as dv_in:
    dv = pkl.load(dv_in)
# read the model
with open(model_file,'rb') as model_in:
    model = pkl.load(model_in)


app = Flask('credit')
@app.route('/predict', methods=['POST'])

def predict():
    """
    Fucntion to predict if the credit is approved or not
    """
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    credit = y_pred >= 0.5

    result = {
        'credit_probability': float(y_pred),
        'credit': bool(credit)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)