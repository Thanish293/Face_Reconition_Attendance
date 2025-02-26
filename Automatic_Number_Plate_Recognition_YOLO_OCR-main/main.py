# Importing the necessary libraries for the program to run.
from ai.ai_model import load_yolov5_model
from ai.ai_model import detection
from helper.params import Parameters
from helper.general_utils import filter_text
from helper.general_utils import save_results
from ai.ocr_model import easyocr_model_load
from ai.ocr_model import easyocr_model_works
from utils.visual_utils import *
import cv2
from datetime import datetime
import time  # For adding delay instead of using waitKey

# Loading the parameters from the params.py file.
params = Parameters()

if __name__ == "__main__":
    print("Starting ANPR system...")
    # Loading the model and labels from the ai_model.py file.
    model, labels = load_yolov5_model()
    # Capturing the video from the webcam.
    camera = cv2.VideoCapture(0)
    # Loading the model for the OCR.
    text_reader = easyocr_model_load()
    
    print("Press Ctrl+C to exit the program")
    
    try:
        while True:
            # Reading the video from the webcam.
            ret, frame = camera.read()
            if ret:
                try:
                    # Detecting the text from the image.
                    detected, coords = detection(frame, model, labels)
                    # Reading the text from the image.
                    resulteasyocr = text_reader.readtext(
                        detected
                    )  # text_read.recognize() , you can use cropped plate image or whole image
                    # Filtering the text from the image.
                    text = filter_text(params.rect_size, resulteasyocr, params.region_threshold)
                    print("Detected text:", text)
                    
                    # Only save results if text was detected
                    if text and len(text) > 0:
                        save_results(text[-1], "ocr_results.csv", "Detection_Images")
                        print(f"License plate detected: {text[-1]}")
                    
                except Exception as e:
                    print(f"Error in processing frame: {str(e)}")
            
            # Use a small delay instead of waitKey
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        # Clean up
        camera.release()
        print("Camera released")