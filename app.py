import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(docs_url="/docs")

MODEL_PATH = "model_transfer.keras"
IMG_SIZE = 224

# Carrega modelo uma vez só ao iniciar o servidor
model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/", include_in_schema=False)
async def interface():
    return FileResponse("index.html")


@app.get("/health_check")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Nao foi possivel ler a imagem.")

    image = np.array(image)

    # Preprocessamento da imagem
    x = tf.convert_to_tensor(image, dtype=tf.float32)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = tf.expand_dims(x, axis=0)

    # Inferência
    y = model.predict(x, verbose=0)

    prob = float(y[0][0])
    pred = 1 if prob >= 0.5 else 0
    label = "dog" if pred == 1 else "cat"

    return {
        "prediction": label,
        "probability": prob
    }