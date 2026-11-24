from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np


app = FastAPI(docs_url="/", title="API de Teste", description="API de Teste para estudo do FastAPI", version="1.0.0")

IMG_SIZE = 224

#Carregar o modelo treinado
model = tf.keras.models.load_model("model_transfer.keras")

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    # Ler a imagem enviada pelo usuário
    image = image.open(file.file).convert("RGB")
    image = np.array(image)
    # Pre-processar a imagem para o tamanho esperado pelo modelo
    x = tf.convert_to_tensor(image, dtype=tf.float32)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = tf.expand_dims(x, axis=0)
    #Inferir a imagem usando o modelo carregado
    y = model.predict(x, verbose=0)

    prob = float(y[0][0])
    pred = 1 if prob >= 0.5 else 0
    label = "dog" if pred == 1 else "cat"

    return {
        "prediction": label,
        "probability": prob
    }