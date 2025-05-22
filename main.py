from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

app = FastAPI()
reader = easyocr.Reader(['en'], gpu=False)

def get_dominant_color(image):
    image = cv2.resize(image, (100, 100))
    image = image.reshape((-1, 3))
    clt = KMeans(n_clusters=1)
    clt.fit(image)
    dominant = clt.cluster_centers_[0]
    return tuple(map(int, dominant))

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as f:
        shutil.copyfileobj(file.file, f)

    image_path = "temp.jpg"
    image = cv2.imread(image_path)

    # License Plate
    results = reader.readtext(image_path)
    plate = results[0][1] if results else "Not Detected"

    # Stub for Make/Model
    make_model = "Toyota Camry"

    # Color
    rgb = get_dominant_color(image)
    color = f"RGB{rgb}"

    return JSONResponse({
        "make": make_model,
        "color": color,
        "license_plate": plate
    })
