import os
import pandas as pd
from ultralytics import YOLO
import cv2

# Training
def train_yolov8(data_path, model_path, trained_model_path, output_dir):
    model = YOLO(model_path)  # load model
    
    epochs = 10
    imgsz = 640
    batch = 16
    
    model.train(
    data=data_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    project=output_dir,
    name='tomato',
    cache=True
    )
    model.save(trained_model_path)
    print(f"Model trained and saved to {trained_model_path}")

def validate_yolov8(model_path, data_path):
    model = YOLO(model_path)
    imgsz=640
    results = model.val(data=data_path, imgsz=imgsz)
    print("Validation Results:")
    print(results)

def test_yolov8(model_path, source, output_image_path):
    model = YOLO(model_path)
    imgsz=640
    results = model.predict(source=source, imgsz=imgsz)
    print("Test Results:")
    for result in results:
        print(result)
    result_img = results[0].plot()

    cv2.imwrite(output_image_path, result_img)
    print(f"Result image saved to {output_image_path}")

if __name__=='__main__':
    # Paths to your dataset and models
    data_config = 'C:\\Users\\Kim\\Desktop\\GitHub\\personal-project-2\\data.yaml'
    initial_model = 'C:\\Users\\Kim\\Desktop\\GitHub\\personal-project-2\\yolov8n.pt'  # Or any other YOLOv8 pre-trained model
    final_model = 'C:\\Users\\Kim\\Desktop\\GitHub\\personal-project-2\\yolov8n_trained.pt'
    output_dir = 'C:\\Users\\Kim\\Desktop\\GitHub\\personal-project-2\\runs\\train'
    output_image_path = 'C:\\Users\\Kim\\Desktop\\GitHub\\personal-project-2\\test_result\\riped_tomato_97.jpeg'

    # Train the model
    #train_yolov8(data_config, initial_model, final_model, output_dir)

    # Validate the model
    #validate_yolov8(final_model, data_config)

    # Test the model on a new image or video
    test_yolov8(final_model, 'C:\\Users\\Kim\\Desktop\\GitHub\\personal-project-2\\dataset\\test\\images\\riped_tomato_97.jpeg', output_image_path)