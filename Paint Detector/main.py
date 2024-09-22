import os
import io
import base64
import pandas as pd
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the CSV file for color matching
df = pd.read_csv('colors.csv')

# Assuming df["RGB"] is a column containing RGB tuples
df["RGB"] = df["RGB"].apply(lambda x: eval(x))  # Convert string to tuple if necessary

# Prepare data for KNN (convert RGB tuples into a list of lists)
X = df["RGB"].tolist()
y = df.index.tolist()  # Dummy target (index)

# Train KNN model with k=1 (closest neighbor)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file is part of the request
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # Check if the user selected a file
        if file.filename == '':
            return "No selected file"
        
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Load the saved image using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                return "Error loading image"

            # Convert the image from BGR to RGB
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Crop the image by 30% from each side
            height, width, _ = image_rgb.shape
            x_crop = int(width * 0.25)  # Crop 20% from each side horizontally
            y_crop = int(height * 0.25)  # Crop 20% from each side vertically
            cropped_img = image_rgb[y_crop:height - y_crop, x_crop:width - x_crop]

            # Flatten the image to get a list of RGB values
            rgb_values = cropped_img.reshape(-1, 3)

            # Find the closest matching RGB value using KNN
            closest_colors = []
            for rgb in rgb_values:
                # Predict the closest RGB index
                closest_index = knn.predict([rgb])[0]
                # Get the actual RGB value from the DataFrame
                closest_rgb = df["RGB"].iloc[closest_index]
                closest_colors.append(closest_rgb)

            if not closest_colors:
                return "No matching RGB values found."

            # Find the most frequent color in the cropped image
            sr = pd.Series(closest_colors)
            most_common_rgb = sr.value_counts().idxmax()

            # Get the color name associated with this RGB value
            color_label = df[df["RGB"] == most_common_rgb]["Color Label"].values[0]

            # Plot the RGB color using Matplotlib and save to buffer
            fig, ax = plt.subplots(figsize=(20, 20))  # 2mm box size
            ax.imshow([[most_common_rgb]])
            plt.axis('off')

            # Save the figure to a BytesIO buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # Encode the buffer as base64 and render it in HTML
            color_image = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Redirect to the page to display the result
            return render_template(
                "index.html",
                color_image=color_image,
                color_label=color_label,
                file_url=url_for('static', filename='uploads/' + file.filename)
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
