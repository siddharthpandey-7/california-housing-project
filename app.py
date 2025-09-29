import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling_new.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # uses request.form to get the data from the HTML
    data = [float(x) for x in request.form.values()]
    # data for the final prediction
    final_user_values = np.array(data).reshape(1, -1)
    transformed_user_values = scaler.transform(final_user_values)
    model_prediction = model.predict(transformed_user_values)

    return jsonify(model_prediction[0])

# driver code
if __name__ == "__main__":
    app.run(debug=True)
    
