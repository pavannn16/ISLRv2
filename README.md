# Isolated Sign Language Recognition App

This Streamlit app captures ASL (American Sign Language) signs using a webcam and predicts the corresponding sign using a TensorFlow Lite model. It visualizes the captured sign and provides real-time predictions with an animated view of the captured landmarks for better visualization.

## Installation and Setup

To run this app locally, you'll need to install the necessary packages and set up the environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/pavannn29/ISLRv2.git
   cd ISLRv2-main
   
2.Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Before Running the Script be sure to change the following paths to your local machine path:
   ```bash
   dummy_parquet_skel_file = '/Users/pavan/GIT/ISLRv2/data/239181.parquet'
   tflite_model = '/Users/pavan/GIT/ISLRv2/models/asl_model.tflite'
   csv_file ='/Users/pavan/GIT/ISLRv2/data/train.csv'
   captured_parquet_file = '/Users/pavan/GIT/ISLRv2/captured.parquet'
   ```
PS:edit here in the app.py code:
<img width="775" alt="image" src="https://github.com/pavannn29/ISLRv2/assets/104923000/a7834283-481e-436d-a7d8-1e7a4551ba58">



Run the Streamlit app:
  ```bash
   streamlit run app.py
  ```
  
Set the duration (in seconds) for capturing the ASL sign.

Click the "Predict Sign" button to capture the sign and receive a prediction.

The captured sign and prediction will be displayed.

## Demo

# Streamlit Interface 
<img width="1434" alt="image" src="https://github.com/pavannn29/ISLRv2/assets/104923000/bbae4483-8fbd-4ce5-b040-f9d4b78fa00d">

# Setting the Duration
<img width="1434" alt="image" src="https://github.com/pavannn29/ISLRv2/assets/104923000/73d361d1-9a79-4912-8f3c-6b33aa78320c">

# Realtime Prediction
<img width="1434" alt="image" src="https://github.com/pavannn29/ISLRv2/assets/104923000/951e4224-3a73-43a8-8c83-483b19c35a84">
<img width="1434" alt="image" src="https://github.com/pavannn29/ISLRv2/assets/104923000/9fef09da-e5ec-46e5-b800-8d0293a6a163">










