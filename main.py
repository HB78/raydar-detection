from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import os
from PIL import Image
from ultralytics import YOLO

# Initialisation
app = FastAPI(title="Raydar Detection API")

# CORS - permet à ton API Next.js d'appeler ce serveur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, mets l'URL de ton API Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle YOLO au démarrage
# On utilise yolov8n (nano) pour commencer - rapide et léger
# Tu pourras upgrader vers yolov8m ou yolov8l plus tard
model = YOLO("yolov8n.pt")

class ImageRequest(BaseModel):
    image_base64: str  # Image encodée en base64

class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

class DetectionResponse(BaseModel):
    success: bool
    detections: list[Detection]
    crops_base64: list[str]  # Images croppées en base64 pour GPT-4o

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Raydar Detection API is running"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_products(request: ImageRequest):
    try:
        # 1. Décoder l'image base64
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # 2. Lancer la détection YOLO
        results = model(image, conf=0.3)  # conf = seuil de confiance minimum
        
        detections = []
        crops_base64 = []
        
        # 3. Parcourir les résultats
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Coordonnées de la bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # On garde seulement les objets pertinents (bouteilles, etc.)
                # YOLO détecte beaucoup de choses, on filtre
                relevant_classes = ["bottle", "cup", "bowl", "banana", "apple", "sandwich", "orange", "carrot", "cell phone", "book"]
                
                if class_name in relevant_classes or confidence > 0.5:
                    detections.append(Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        class_name=class_name
                    ))
                    
                    # 4. Cropper l'image pour ce produit
                    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    # Convertir le crop en base64
                    buffer = io.BytesIO()
                    crop.save(buffer, format="JPEG", quality=85)
                    crop_base64 = base64.b64encode(buffer.getvalue()).decode()
                    crops_base64.append(crop_base64)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            crops_base64=crops_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pour Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)