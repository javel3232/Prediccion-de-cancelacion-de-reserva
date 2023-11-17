from fastapi import HTTPException, FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from scripts.train import train_and_save_model

app = FastAPI()

# Cargar el modelo al inicio de la aplicación
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
    
@app.post("/predict/")
async def predict(data: dict):
    # Asegurarse de que los datos tengan la estructura adecuada
    if 'id' not in data:
        raise HTTPException(status_code=400, detail="Falta el campo 'id' en los datos de entrada.")

    input_data = pd.DataFrame(data, index=[0])
    X = input_data.drop('id', axis=1)

    prediction = model.predict(X)
    prediction_result = prediction.tolist()  # Convertir a una lista si es necesario

    return JSONResponse(content={"prediction": prediction_result})


# Train endpoint
@app.post("/train")
async def train_endpoint():
    try:
        # Llama a la función de entrenamiento y guarda el modelo con un nuevo nombre
        trained_model = train_and_save_model(model_file='modelNew.pkl')
        
        if trained_model:
            return JSONResponse(content={"message": "Model trained and saved successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to train and save model. Check logs for details.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
