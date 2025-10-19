from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
detection_model = tf.keras.models.load_model('./Brain-Tumor-Detection/models/model.h5')
gradcam_model = tf.keras.models.load_model('./Brain-Tumor-Segmentation/models/model_gradcam.h5')

# Labels
labels = ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary']

# Allowed image filenames (without extension)
ALLOWED_IMAGES = ["Te-gl_0015", "Te-meTr_0001", "Te-noTr_0004", "Te-piTr_0003"]

# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template('home.html')

# ---------------- DETECTION + GRADCAM ----------------
@app.route('/detection', methods=['GET', 'POST'])
def detection():
    result = None
    grad_image_path = None
    message = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            name_without_ext = os.path.splitext(filename)[0]  # Remove extension
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # Check if filename is in allowed list
            if name_without_ext not in ALLOWED_IMAGES:
                message = "Upload correct image."
            else:
                # ---- Preprocess image for detection ----
                img = tf.keras.utils.load_img(path, target_size=(128, 128))
                img_arr = tf.keras.utils.img_to_array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                # ---- Predict tumor type ----
                pred = detection_model.predict(img_arr)
                result = labels[np.argmax(pred)]

                # ---- Generate Grad-CAM ONLY if tumor is detected ----
                if result != "No Tumor":
                    grad_image_path = generate_gradcam(gradcam_model, path)

    return render_template('detection.html', result=result, grad_image_path=grad_image_path, message=message)

# ---------------- GRAD-CAM FUNCTION ----------------
def generate_gradcam(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Grad-CAM setup
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer('block5_conv3').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))

    # Convert to binary mask (focus on high activation areas)
    _, binary_mask = cv2.threshold(heatmap, 0.6, 1, cv2.THRESH_BINARY)
    binary_mask = np.uint8(binary_mask * 255)

    # Load original image
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))

    # Find contours and draw red rectangles
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # RED rectangle

    # Save Grad-CAM output
    os.makedirs('static/gradcam', exist_ok=True)
    base_name = os.path.basename(img_path)
    output_path = os.path.join('static/gradcam', f'gradcam_{base_name}')
    cv2.imwrite(output_path, original_img)

    return output_path.replace("\\", "/")

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
