# import cv2
# import numpy as np
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image

# # Load the pre-trained InceptionV3 model
# model = InceptionV3(weights='imagenet')

# def predict_food_from_frame(frame):
#     try:
#         # Resize the frame to the required input shape for InceptionV3 (299x299)
#         frame_resized = cv2.resize(frame, (299, 299))
#         x = image.img_to_array(frame_resized)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)

#         # Make predictions
#         preds = model.predict(x)
#         decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

#         # Print the top predictions
#         print("Predictions:")
#         for _, category, confidence in decoded_preds:
#             print(f"{category}: {confidence:.2f}")

#         # Display the frame
#         cv2.imshow('Camera', frame)
#         cv2.waitKey(1)  # Keep the window open

#     except Exception as e:
#         print(f"Error: {e}. Please check the camera feed.")

# # Open the default camera (usually 0)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()  # Read a frame from the camera
#     if not ret:
#         break
    
#     predict_food_from_frame(frame)

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

def predict_food_from_frame(frame, confidence_threshold=0.7):
    try:
        # Resize the frame to the required input shape for InceptionV3 (299x299)
        frame_resized = cv2.resize(frame, (299, 299))
        x = image.img_to_array(frame_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

        # Print the top predictions
        print("Predictions:")
        for _, category, confidence in decoded_preds:
            print(f"{category}: {confidence:.2f}")
            if confidence >= confidence_threshold and "food" in category.lower():
                return True

        # Display the frame
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)  # Keep the window open

    except Exception as e:
        print(f"Error: {e}. Please check the camera feed.")

# Open the default camera (usually 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break
    
    food_detected = predict_food_from_frame(frame)
    if food_detected:
        break

cap.release()
cv2.destroyAllWindows()
