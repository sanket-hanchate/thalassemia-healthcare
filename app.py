from flask import Flask, render_template, request
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
import numpy as np
import joblib

# Define your model architecture
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = NNet()
model.load_state_dict(torch.load('wellness_model.pt'))
model.eval()

# Load scaler if used
with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

app = Flask(__name__)
app.secret_key = "sanket_hanchate"

# MongoDB Config
app.config["MONGO_URI"] = "mongodb://localhost:27017/health_db"
mongo = PyMongo(app)
users = mongo.db.users  # MongoDB collection

@app.route("/signup", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_data = {
            "name": request.form.get("name"),
            "age": request.form.get("age"),
            "gender": request.form.get("gender"),
            "role": request.form.get("role"),
            "blood_group": request.form.get("blood_group"),
            "location": request.form.get("location"),
            "email": request.form.get("email"),
            "phone": request.form.get("phone"),
            "password": generate_password_hash(request.form.get("password")),
            "last_donation": request.form.get("last_donation"),
            "thal_type": request.form.get("thal_type")
        }
        users.insert_one(user_data)
        flash("Registration successful!", "success")
        return redirect("/")
    return render_template("signup.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            session["username"] = username
            flash("Logged in successfully!")
            return redirect(url_for("dashboard"))  # redirect to dashboard
        else:
            flash("Invalid credentials. Try again.")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template("dashboard.html", username=session['username'])
    else:
        return redirect(url_for("login"))

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("home"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/wellness')
def wellness():
    return render_template('wellness.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['hours_of_sleep']),
            float(request.form['stress_level']),
            float(request.form['daily_calorie_intake']),
            float(request.form['active_heart_rate']),
            float(request.form['resting_heart_rate']),
        ]

        X_scaled = scaler.transform([features])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        output = model(X_tensor)
        prediction = (output >= 0.5).float().item()

        result = "Well Rested ✅" if prediction == 1 else "Not Well Rested ❌"

        return render_template('wellness.html', prediction=result)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
