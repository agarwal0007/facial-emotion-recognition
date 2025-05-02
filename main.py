from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os, uuid
from lab import predict_sequence, load_model
import torch

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("best_vit_sequence_model.pth", device)

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    folder = f"temp_{uuid.uuid4()}"
    os.makedirs(folder, exist_ok=True)

    try:
        for file in files:
            file_path = os.path.join(folder, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        prediction = predict_sequence(folder, model=model, device=device)

        # Optional: Map to label name
        class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        return {"prediction": prediction, "label": class_names[prediction]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        shutil.rmtree(folder)
