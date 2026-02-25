from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.preprocessing import Preprocessor
from models.inference import DeepfakeDetector
from utils.evidence import EvidenceManager
from utils.crypto import SecurityManager

# Setup Database
SQLALCHEMY_DATABASE_URL = "sqlite:///./phoenix.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MediaAnalysis(Base):
    __tablename__ = "media_analysis"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    file_hash = Column(String)
    status = Column(String) # processing, completed, failed
    score = Column(Float, nullable=True)
    visual_score = Column(Float, nullable=True)
    audio_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    evidence_bundle_path = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Project Phoenix API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Managers
preprocessor = Preprocessor()
detector = DeepfakeDetector()
evidence_manager = EvidenceManager()
security_manager = SecurityManager()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/analyze")
async def analyze_media(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    analysis_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{analysis_id}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Calculate Hash
    file_hash = evidence_manager.calculate_sha256(temp_path)
    file_info = {
        "filename": file.filename,
        "hash": file_hash,
        "type": file.content_type,
        "size": os.path.getsize(temp_path)
    }
    
    # Save Initial State
    db = SessionLocal()
    db_analysis = MediaAnalysis(
        id=analysis_id,
        filename=file.filename,
        file_hash=file_hash,
        status="processing"
    )
    db.add(db_analysis)
    db.commit()
    db.close()
    
    # Run Inference in Background
    background_tasks.add_task(run_inference, analysis_id, temp_path, file_info)
    
    return {"analysis_id": analysis_id, "status": "processing"}

async def run_inference(analysis_id: str, file_path: str, file_info: dict):
    db = SessionLocal()
    db_analysis = db.query(MediaAnalysis).filter(MediaAnalysis.id == analysis_id).first()
    
    try:
        visual_score = None
        audio_score = None
        
        # Preprocessing & Inference
        if file_info["type"].startswith("video"):
            frames = preprocessor.extract_frames(file_path)
            visual_score = detector.predict_visual(frames)
            # Extracted audio could be passed here in a real scenario
            # audio_path = extract_audio(file_path)
            # audio_score = detector.predict_audio(preprocessor.generate_spectrogram(audio_path))
        elif file_info["type"].startswith("image"):
            img = preprocessor.preprocess_image(file_path)
            visual_score = detector.predict_visual(np.expand_dims(img, axis=0))
        elif file_info["type"].startswith("audio"):
            spec = preprocessor.generate_spectrogram(file_path)
            audio_score = detector.predict_audio(spec)
            
        fused_score = detector.fuse_scores(visual_score, audio_score)
        
        # Evidence Generation
        analysis_results = {
            "score": detector.get_authenticity_percentage(fused_score),
            "visual_score": detector.get_authenticity_percentage(visual_score) if visual_score else None,
            "audio_score": detector.get_authenticity_percentage(audio_score) if audio_score else None,
            "status": "completed"
        }
        
        manifest, manifest_path = evidence_manager.create_manifest(file_info, analysis_results)
        
        # PDF Report
        report_path = os.path.join("manifests", f"{analysis_id}_report.pdf")
        evidence_manager.generate_pdf_report(manifest, report_path)
        
        # Security: Sign Manifest
        sig_path = security_manager.sign_manifest(manifest_path)
        
        # Bundle Evidence
        bundle_path = os.path.join("manifests", f"{analysis_id}_evidence")
        final_bundle = security_manager.encrypt_bundle([manifest_path, report_path, sig_path], bundle_path)
        
        # Update DB
        db_analysis.status = "completed"
        db_analysis.score = analysis_results["score"]
        db_analysis.visual_score = analysis_results["visual_score"]
        db_analysis.audio_score = analysis_results["audio_score"]
        db_analysis.evidence_bundle_path = final_bundle
        db.commit()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        db_analysis.status = "failed"
        db.commit()
    finally:
        db.close()
        # In a real app, delete the original upload after processing or bundle creation
        # os.remove(file_path)

@app.get("/score/{analysis_id}")
async def get_score(analysis_id: str):
    db = SessionLocal()
    analysis = db.query(MediaAnalysis).filter(MediaAnalysis.id == analysis_id).first()
    db.close()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    return {
        "id": analysis.id,
        "status": analysis.status,
        "score": analysis.score,
        "visual_score": analysis.visual_score,
        "audio_score": analysis.audio_score,
        "timestamp": analysis.created_at
    }

@app.get("/report/{analysis_id}")
async def get_report(analysis_id: str):
    db = SessionLocal()
    analysis = db.query(MediaAnalysis).filter(MediaAnalysis.id == analysis_id).first()
    db.close()
    
    if not analysis or analysis.status != "completed":
        raise HTTPException(status_code=404, detail="Report not ready or analysis failed")
        
    from fastapi.responses import FileResponse
    return FileResponse(analysis.evidence_bundle_path, media_type='application/zip', filename=f"Phoenix_Evidence_{analysis.id}.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
