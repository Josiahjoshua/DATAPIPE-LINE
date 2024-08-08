from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import pydicom
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'breast_cancer_pipeline',
    default_args=default_args,
    description='Breast Cancer Data Pipeline',
    schedule_interval=timedelta(days=30),
)

image_dir = '/path/to/images'  # Adjust path if necessary
demographic_file = '/path/to/demographic_data.csv'  # Adjust path if necessary
model_path = 'E:/Model_output/AfyaAI_model_Resnet50.keras'  # Path to your trained model

# Load the trained model
def load_model_task():
    model = load_model(model_path)
    return model

# Load images and preprocess them
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(image_dir, filename))
            img_array = ds.pixel_array
            img_array = cv2.resize(img_array, (50, 50))
            img_array = np.stack((img_array,)*3, axis=-1)  # Convert to 3 channels
            img_array = img_array / 255.0  # Normalize pixel values
            images.append({'filename': filename, 'image': img_array, 'type': 'dcm'})
        else:
            print(f"Skipping non-DICOM file: {filename}")
    return images

# Classify images using the model
def classify_images(images, model):
    for img in images:
        if img['type'] == 'dcm':
            img_batch = np.expand_dims(img['image'], axis=0)
            predictions = model.predict(img_batch)
            img['prediction'] = predictions[0].argmax()
            img['category'] = 'mammogram'
        else:
            img['category'] = 'other'
    return images

# Merge image data with demographic data
def merge_data(images, demographic_data):
    merged_data = []
    for img in images:
        filename = img['filename']
        patient_data = demographic_data[demographic_data['filename'] == filename]
        if not patient_data.empty:
            data = {**img, **patient_data.to_dict(orient='records')[0]}
            merged_data.append(data)
    return merged_data

# Save the processed data
def save_processed_data(merged_data):
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(merged_data, f)

# Retrain the model with new images and labels
def retrain_model(images, labels):
    model = load_model(model_path)

    # Convert images and labels to numpy arrays
    images = np.array([img['image'] for img in images])
    labels = np.array(labels)

    # One-hot encode the labels if needed
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)

    # Define the training process
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.25)

    # Save the retrained model
    model.save(model_path)

    return model

def task_load_images():
    images = load_images(image_dir)
    return images

def task_classify_images(**context):
    images = context['task_instance'].xcom_pull(task_ids='load_images')
    model = context['task_instance'].xcom_pull(task_ids='load_model_task')
    classified_images = classify_images(images, model)
    return classified_images

def task_merge_data(**context):
    classified_images = context['task_instance'].xcom_pull(task_ids='classify_images')
    demographic_data = pd.read_csv(demographic_file)
    merged_data = merge_data(classified_images, demographic_data)
    return merged_data

def task_save_processed_data(**context):
    merged_data = context['task_instance'].xcom_pull(task_ids='merge_data')
    save_processed_data(merged_data)

def task_retrain_model(**context):
    images = context['task_instance'].xcom_pull(task_ids='load_images')
    labels = context['task_instance'].xcom_pull(task_ids='load_labels')  # Assuming you have a load_labels task
    retrained_model = retrain_model(images, labels)
    return retrained_model

# Define tasks
t0 = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_task,
    dag=dag,
)

t1 = PythonOperator(
    task_id='load_images',
    python_callable=task_load_images,
    dag=dag,
)

t2 = PythonOperator(
    task_id='classify_images',
    python_callable=task_classify_images,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='merge_data',
    python_callable=task_merge_data,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='save_processed_data',
    python_callable=task_save_processed_data,
    provide_context=True,
    dag=dag,
)

t5 = PythonOperator(
    task_id='retrain_model',
    python_callable=task_retrain_model,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
t0 >> t1 >> t2 >> t3 >> t4 >> t5
