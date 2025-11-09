from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load the trained model
model = pickle.load(open('model/random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)

        prediction = model.predict(final_input)[0]
        result_text = "✅ Loan Approved (Prediction = 1)" if prediction == 1 else "❌ Loan Rejected (Prediction = 0)"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
