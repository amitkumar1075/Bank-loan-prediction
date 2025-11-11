from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
import os

model_path = os.path.join(os.path.dirname(__file__),"model", r"random_forest_model.joblib")
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all 16 features from form
        features = [
            float(request.form['age']),
            float(request.form['job']),
            float(request.form['marital']),
            float(request.form['education']),
            float(request.form['fault']),
            float(request.form['balance']),
            float(request.form['housing']),
            float(request.form['loan']),
            float(request.form['contact']),
            float(request.form['day']),
            float(request.form['month']),
            float(request.form['duration']),
            float(request.form['campaign']),
            float(request.form['pdays']),
            float(request.form['previous']),
            float(request.form['poutcome'])
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]

        result = "✅ Customer will subscribe!" if prediction == 1 else "❌ Customer will not subscribe."
        return render_template('index.html', result_text=result)

    except Exception as e:
        return render_template('index.html', result_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
