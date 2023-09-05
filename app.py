from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained Neural Network model
loaded_model = load_model('customer_churn_nn_model.h5')

# Initialize the label encoder for categorical variables
label_encoder = LabelEncoder()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        # Get input data from the form
        age = int(request.form['Age'])
        gender = request.form['Gender']
        location = request.form['Location']
        subscription_length = int(request.form['Subscription_Length_Months'])
        monthly_bill = float(request.form['Monthly_Bill'])
        total_usage_gb = int(request.form['Total_Usage_GB'])

        # Create a DataFrame with the user's input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Location': [location],
            'Subscription_Length_Months': [subscription_length],
            'Monthly_Bill': [monthly_bill],
            'Total_Usage_GB': [total_usage_gb]
        })

        # Fit and transform categorical variables using the label encoder
        categorical_cols = ['Gender', 'Location'] 
        for col in categorical_cols:
            label_encoder.fit(input_data[col])
            input_data[col] = label_encoder.transform(input_data[col])

        # Perform feature scaling (standardization)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(input_data)

        # Make predictions using the loaded model
        churn_probability = loaded_model.predict(scaled_data)[0][0]

        return render_template('result.html', churn_probability=churn_probability)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)



