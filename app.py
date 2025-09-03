from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = int(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100  # Survival probability %

        result_text = "Survived" if prediction == 1 else "Not Survived"
        return render_template("result.html", result=result_text, probability=round(probability, 2))

    except Exception as e:
        return render_template("result.html", result=f"Error: {str(e)}", probability=None)

if __name__ == "__main__":
    app.run(debug=True, port=6060)
