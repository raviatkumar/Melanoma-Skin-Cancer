from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException  # Import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import shutil
import numpy as np
import tensorflow as tf

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define class labels
class_labels = ["benign", "malignant"]

# Create a temporary directory for uploaded images
if not os.path.exists("temp"):
    os.makedirs("temp")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        # Save the uploaded image to the temporary directory
        with open(f"temp/{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(f"temp/{image.filename}", target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values to [0, 1]

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]

        # Remove the temporary image file
        os.remove(f"temp/{image.filename}")

        return templates.TemplateResponse("result.html", {"request": request, "prediction": predicted_label, "image": f"/temp/{image.filename}"})
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
