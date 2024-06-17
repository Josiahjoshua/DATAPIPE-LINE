from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import pydicom
from PIL import Image
import pickle

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

def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(image_dir, filename))
            images.append({'filename': filename, 'image': ds.pixel_array, 'type': 'dcm'})
        else:
            # Log or print a message indicating the file is being skipped
            print(f"Skipping non-DICOM file: {filename}")
    return images

def classify_images(images):
    for img in images:
        if img['type'] == 'dcm':
            img['category'] = 'mammogram'
        else:
            img['category'] = 'other'
    return images

def merge_data(images, demographic_data):
    merged_data = []
    for img in images:
        filename = img['filename']
        patient_data = demographic_data[demographic_data['filename'] == filename]
        if not patient_data.empty:
            data = {**img, **patient_data.to_dict(orient='records')[0]}
            merged_data.append(data)
    return merged_data

def save_processed_data(merged_data):
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(merged_data, f)

def task_load_images():
    images = load_images(image_dir)
    return images

def task_classify_images(**context):
    images = context['task_instance'].xcom_pull(task_ids='load_images')
    classified_images = classify_images(images)
    return classified_images

def task_merge_data(**context):
    classified_images = context['task_instance'].xcom_pull(task_ids='classify_images')
    demographic_data = pd.read_csv(demographic_file)
    merged_data = merge_data(classified_images, demographic_data)
    return merged_data

def task_save_processed_data(**context):
    merged_data = context['task_instance'].xcom_pull(task_ids='merge_data')
    save_processed_data(merged_data)

# Define tasks
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

# Set task dependencies
t1 >> t2 >> t3 >> t4
