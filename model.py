import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import time
import os
from skimage.feature import hog
from skimage import exposure
from scipy import ndimage

# Function to download and prepare the MNIST dataset with additional features
def prepare_mnist_dataset():
    print("Starting to prepare MNIST dataset...")
    # Check if the model already exists
    if os.path.exists('enhanced_mnist_model.h5'):
        print("Enhanced model already exists, loading...")
        return keras.models.load_model('enhanced_mnist_model.h5')
    
    # Load MNIST dataset
    print("Downloading and preparing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Reshape the data for the CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Data augmentation with more transformations
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Build the enhanced model with residual connections
    print("Building the enhanced model with residual connections...")
    inputs = keras.Input(shape=(28, 28, 1))
    
    # First block with residual connection
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Second block with residual connection
    x_res = x
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    # Add upsampled residual connection
    x_res = keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same')(x_res)
    x = keras.layers.add([x, x_res])
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Third block
    x = keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    # Dense layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with better optimizer settings
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    # Reduce learning rate when plateau is reached
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )
    
    # Model checkpoint to save the best model
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_mnist_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model with data augmentation
    print("Training the enhanced model...")
    model.fit(datagen.flow(x_train, y_train, batch_size=128),
              epochs=4,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, reduce_lr, checkpoint])
    
    # Load the best model
    model = keras.models.load_model('best_mnist_model.h5')
    
    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Enhanced model accuracy: {accuracy * 100:.2f}%")
    
    # Save the model with a different name
    model.save('enhanced_mnist_model.h5')
    print("Enhanced model saved to enhanced_mnist_model.h5")
    
    return model

# Function to extract HOG features
def extract_hog_features(img):
    # HOG parameters
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    
    hog_features = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=False, block_norm='L2-Hys')
    
    return hog_features

# Advanced preprocessing function
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while reducing noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to get binary image
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate to connect broken parts of digits
    dilated = cv2.dilate(closing, kernel, iterations=1)
    
    return dilated

# Improved function to find and sort contours
def find_digits(preprocessed_img, original_img):
    # Find contours
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size, aspect ratio, and solidity
    digit_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        # If contour is big enough, has reasonable aspect ratio and solidity
        if (w > 15 and h > 25 and w < 150 and h < 150 and 
            0.1 < aspect_ratio < 1.2 and solidity > 0.5):
            # Check if it contains enough "ink" (foreground pixels)
            mask = np.zeros(preprocessed_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            foreground_pixels = cv2.countNonZero(cv2.bitwise_and(preprocessed_img, mask))
            if foreground_pixels > 0.25 * w * h:
                # Add padding to the bounding box
                padding = 5
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(w + 2*padding, preprocessed_img.shape[1] - x_pad)
                h_pad = min(h + 2*padding, preprocessed_img.shape[0] - y_pad)
                
                digit_contours.append((x_pad, y_pad, w_pad, h_pad))
    
    # Sort contours from left to right
    digit_contours.sort(key=lambda x: x[0])
    
    return digit_contours

# Function to predict a single digit
def predict_digit(img, model):
    # Ensure the image is square by adding padding
    h, w = img.shape
    diff = abs(h - w)
    padding = diff // 2
    
    if h > w:
        padded_img = np.pad(img, ((0, 0), (padding, diff - padding)), mode='constant', constant_values=0)
    else:
        padded_img = np.pad(img, ((padding, diff - padding), (0, 0)), mode='constant', constant_values=0)
    
    # Add extra padding around all sides
    extra_padding = 5
    padded_img = np.pad(padded_img, ((extra_padding, extra_padding), (extra_padding, extra_padding)), 
                        mode='constant', constant_values=0)
    
    # Resize to 28x28
    resized = cv2.resize(padded_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Center the digit in the image
    cy, cx = ndimage.center_of_mass(resized)
    rows, cols = resized.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    resized = cv2.warpAffine(resized, M, (cols, rows))
    
    # Normalize and reshape for the model
    normalized = resized.astype('float32') / 255
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    # Make prediction
    predictions = model.predict(reshaped, verbose=0)
    
    # Top 2 predictions for potential confusion resolution
    top_indices = np.argsort(predictions[0])[-2:][::-1]
    top_probabilities = predictions[0][top_indices]
    
    # Return both top predictions if they're close
    if len(top_indices) > 1 and (top_probabilities[0] - top_probabilities[1]) < 0.2:
        return top_indices[0], top_probabilities[0], top_indices[1], top_probabilities[1]
    else:
        return top_indices[0], top_probabilities[0], None, None

# Enhanced main function with multi-digit grouping
def main():
    print("Starting the enhanced handwritten digit recognition program...")
    try:
        # Prepare the model
        print("Loading or training the enhanced model...")
        model = prepare_mnist_dataset()
        
        # Initialize the webcam
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        print("Webcam initialized successfully!")
        
        # Set the width and height of the capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Region of interest parameters
        roi_x, roi_y, roi_width, roi_height = 100, 100, 440, 280
        
        last_prediction_time = time.time()
        last_number = None
        prediction_history = []  # Keep track of recent predictions
        
        print("Starting the main loop. Press 'q' to quit.")
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip the frame for a more intuitive experience
            frame = cv2.flip(frame, 1)
            
            # Draw the ROI
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
            
            # Extract the ROI
            roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            
            # Every half second, process the ROI to detect digits
            current_time = time.time()
            if current_time - last_prediction_time > 0.5:
                # Preprocess the ROI
                processed_roi = preprocess_image(roi)
                
                # Display the processed ROI (for debugging)
                processed_display = cv2.resize(processed_roi, (220, 140))
                frame[10:150, 400:620] = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, "Processed", (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Find potential digits
                digit_contours = find_digits(processed_roi, roi)
                
                # Process and predict each digit
                digits = []
                confidences = []
                for i, (x, y, w, h) in enumerate(digit_contours):
                    # Extract the digit
                    digit_img = processed_roi[y:y + h, x:x + w]
                    
                    # Skip if the digit is too small or too large
                    if w * h < 400 or w * h > 10000:
                        continue
                    
                    # Predict the digit
                    digit1, conf1, digit2, conf2 = predict_digit(digit_img, model)
                    
                    # Only use predictions with high confidence
                    if conf1 > 0.6:
                        digits.append((digit1, x))  # Store digit and x-coordinate
                        confidences.append(conf1)
                        
                        # Display the digit and confidence on the frame
                        color = (0, int(255 * conf1), int(255 * (1 - conf1)))
                        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(roi, str(digit1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                        # Display alternative prediction if available and close
                        if digit2 is not None and conf2 > 0.3:
                            cv2.putText(roi, f"or {digit2}", (x, y + h + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 150), 1)
                
                # Sort digits by x-coordinate
                digits.sort(key=lambda x: x[1])
                
                # Group digits that are horizontally aligned
                grouped_digits = []
                current_group = []
                
                for i, (digit, x) in enumerate(digits):
                    if i == 0 or x - digits[i-1][1] < 50:  # If close to previous digit
                        current_group.append(digit)
                    else:
                        if current_group:
                            grouped_digits.append(current_group)
                        current_group = [digit]
                
                if current_group:
                    grouped_digits.append(current_group)
                
                # Convert the groups to numbers
                numbers = []
                for group in grouped_digits:
                    if group:
                        number = int(''.join(map(str, group)))
                        if number <= 999:  # Only show numbers <= 999
                            numbers.append(number)
                
                # Get the most reliable number from the detected ones
                if numbers:
                    # Update prediction history
                    prediction_history.append(numbers[0] if len(numbers) > 0 else None)
                    # Keep only last 5 predictions
                    if len(prediction_history) > 5:
                        prediction_history.pop(0)
                    
                    # Find most common prediction in history
                    if prediction_history.count(prediction_history[-1]) >= 3:
                        last_number = prediction_history[-1]
                
                last_prediction_time = current_time
            
            # Display the number on the frame
            if last_number is not None:
                cv2.putText(frame, f"Number: {last_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Write digits 0-9 in green box", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Enhanced Handwritten Digit Recognition', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("Cleaning up...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    main()