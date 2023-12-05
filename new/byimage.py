import numpy as np
import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

def predict_food(image_path, confidence_threshold=0.5):
    try:
        # Load the image and resize it to the required input shape for InceptionV3 (299x299)
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

        # Print the top predictions above the confidence threshold
        print("Predictions:")
        for _, category, confidence in decoded_preds:
            if confidence >= confidence_threshold:
                print(f"{category}: {confidence:.2f}")

        # Show the image
        img = cv2.imread(image_path)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}. Please check the image path or format.")

image_path = 'burger.jpg'
predict_food(image_path)
