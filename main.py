"""
DeepFace Backend for Classroom Attendance System
FastAPI server providing face recognition endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import base64
import os
import json
import uuid
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import pandas as pd

# DeepFace for face recognition
from deepface import DeepFace

app = FastAPI(title="Attendance CCTV - DeepFace Backend")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = "student_data"
STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
FACES_DIR = os.path.join(DATA_DIR, "faces")

# DeepFace model - ArcFace is one of the most accurate
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"  # More accurate for various angles
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.68  # ArcFace cosine threshold (same person < 0.68)

# Ensure directories exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "sync_images"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)


class Student(BaseModel):
    id: str
    name: str
    photo_path: str


class RecognitionResult(BaseModel):
    box: dict  # x, y, width, height
    name: str
    confidence: float
    is_known: bool


class RecognizeResponse(BaseModel):
    faces: List[RecognitionResult]


def load_students() -> List[dict]:
    """Load students from JSON file"""
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_students(students: List[dict]):
    """Save students to JSON file"""
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(students, f, indent=2)


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image (BGR)"""
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    
    # Convert to RGB then BGR for OpenCV
    img_array = np.array(img.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/")
async def root():
    return {"status": "ok", "message": "DeepFace Attendance Backend Running"}


@app.get("/students")
@app.get("/api/students")
async def get_students():
    """Get all registered students"""
    students = load_students()
    return {"students": students}


class StudentCreate(BaseModel):
    name: str
    class_name: Optional[str] = None
    course: Optional[str] = None
    photo: Optional[str] = None

@app.post("/students")
@app.post("/api/students/register")
def register_student(data: StudentCreate):
    """Register a new student with their face (optional) via JSON"""
    try:
        student_id = str(uuid.uuid4())
        photo_path = ""
        
        # If photo provided, process with DeepFace
        if data.photo:
            img = base64_to_image(data.photo)
            try:
                faces = DeepFace.extract_faces(
                    img_path=img,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True
                )
                if not faces:
                    raise HTTPException(status_code=400, detail="No face detected in image")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
            
            photo_filename = f"{student_id}.jpg"
            photo_path = os.path.join(FACES_DIR, photo_filename)
            cv2.imwrite(photo_path, img)
            
            # Clear DeepFace cache so find() knows about the new face
            for f in os.listdir(FACES_DIR):
                if f.endswith(".pkl"):
                    os.remove(os.path.join(FACES_DIR, f))
        
        class_val = data.class_name or data.course
        student_record = {
            "id": student_id,
            "name": data.name,
            "class_name": class_val,
            "course": class_val,
            "photo_path": photo_path
        }
        
        # Add to students list
        students = load_students()
        students.append(student_record)
        save_students(students)
        
        return student_record
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/students/{student_id}")
@app.delete("/api/students/{student_id}")
def delete_student(student_id: str):
    """Delete a student"""
    students = load_students()
    student = next((s for s in students if s.get("id") == student_id or s.get("student_id") == student_id or s.get("uuid") == student_id), None)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Remove photo file
    if os.path.exists(student.get("photo_path", "")):
        os.remove(student["photo_path"])
        
    # Clear DeepFace cache after removing face
    for f in os.listdir(FACES_DIR):
        if f.endswith(".pkl"):
            os.remove(os.path.join(FACES_DIR, f))
    
    # Remove from list
    students = [s for s in students if s["id"] != student_id and s.get("student_id") != student_id and s.get("uuid") != student_id]
    save_students(students)
    
    return {"success": True, "deleted_id": student_id}


class StudentUpdate(BaseModel):
    name: str
    course: Optional[str] = None
    class_name: Optional[str] = None

@app.put("/students/{student_id}")
@app.put("/api/students/{student_id}")
def update_student(student_id: str, data: StudentUpdate):
    """Update a student's name and/or class/course"""
    students = load_students()
    student = next((s for s in students if s["id"] == student_id), None)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
        
    student["name"] = data.name
    
    # Store course or class_name mapping
    if data.course is not None:
        student["course"] = data.course
        student["class_name"] = data.course
    elif data.class_name is not None:
        student["class_name"] = data.class_name
        student["course"] = data.class_name
        
    save_students(students)
    return {"success": True, "student": student}


@app.post("/api/recognize")
def recognize_faces(image: str = Form(...)):
    """
    Recognize faces in an image
    Returns bounding boxes and identities for all detected faces
    """
    try:
        img = base64_to_image(image)
        students = load_students()
        
        if not students:
            # no students, just detect
            try:
                faces = DeepFace.extract_faces(
                    img_path=img,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
                results = []
                for face in faces:
                    area = face.get("facial_area", {})
                    results.append({
                        "box": {
                            "x": area.get("x", 0),
                            "y": area.get("y", 0),
                            "width": area.get("w", 0),
                            "height": area.get("h", 0)
                        },
                        "name": "Unknown",
                        "confidence": 0,
                        "is_known": False
                    })
                return {"faces": results}
            except:
                return {"faces": []}
        
        # Use find() directly which does detection + recognition
        # verify() loop is too slow (O(N) detection). find() is O(1) embedding match.
        
        results = []
        try:
             # DeepFace.find returns list of DFs (one per face)
             # IF enforce_detection=False, and no face, returns one empty DF?
             # IF enforce_detection=False, and face found but no match, returns DF (empty usually?)
             # Actually find() logic is a bit complex.
             # To ensure we get ALL faces (even unknown), it's safer to:
             # 1. Detect all faces first (extract_faces)
             # 2. For each face, run find() on the crop
             
             detected_faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
             )
             
             for face_data in detected_faces:
                 area = face_data.get("facial_area", {})
                 box = {
                    "x": area.get("x", 0),
                    "y": area.get("y", 0),
                    "width": area.get("w", 0),
                    "height": area.get("h", 0)
                 }
                 
                 # Skip very small faces
                 if box["width"] < 30 or box["height"] < 30:
                     continue
                 
                 # Get the face crop to pass to find
                 face_img = face_data["face"]
                 
                 # Run find on this specific face
                 # Note: find() expects path or numpy array
                 # face_data["face"] is normalized 0-1, find expects 0-255 int usually if array? 
                 # Actually verify() handles it. find() uses verify internally. 
                 # SAFE BET: Use the original image crop using coordinates, to avoid normalization issues.
                 x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                 face_crop = img[y:y+h, x:x+w]
                 
                 found_match = None
                 confidence = 0
                 
                 try:
                     dfs = DeepFace.find(
                        img_path=face_crop,
                        db_path=FACES_DIR,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR_BACKEND,
                        distance_metric=DISTANCE_METRIC,
                        enforce_detection=False, # already detected
                        silent=True
                     )
                     
                     if len(dfs) > 0 and not dfs[0].empty:
                         # Best match
                         match_path = dfs[0].iloc[0]['identity']
                         match_dist = dfs[0].iloc[0]['distance']
                         
                         filename = os.path.basename(match_path)
                         student_id = os.path.splitext(filename)[0]
                         
                         student = next((s for s in students if s["id"] == student_id), None)
                         if student:
                             found_match = student
                             # Approximate confidence from distance (lower is better)
                             # ArcFace threshold 0.68. 0 is 100%, 0.68 is ~50%? 
                             # Let's map 0..threshold to 100..0 linearly-ish
                             confidence = max(0, min(100, int((1 - match_dist) * 100)))
                 except Exception as e:
                     # print(f"Find error: {e}")
                     pass
                 
                 if found_match:
                     results.append({
                        "box": box,
                        "name": found_match["name"],
                        "confidence": confidence,
                        "is_known": True,
                        "student_id": found_match["id"]
                    })
                 else:
                     results.append({
                        "box": box,
                        "name": "Unknown",
                        "confidence": 0,
                        "is_known": False
                    })

        except Exception as e:
            # Fallback if extract_faces fails entirely
            print(f"Extraction error: {e}")
            return {"faces": []}
            
        return {"faces": results}
    
    except Exception as e:
        print(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Attendance Persistence & Sync ---

ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.json")

class AttendanceRecord(BaseModel):
    id: str  # unique record id
    student_id: str
    student_name: str
    timestamp: str  # ISO format
    method: str  # "live", "manual", "sync"
    synced_at: Optional[str] = None

def load_attendance() -> List[dict]:
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_attendance(records: List[dict]):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(records, f, indent=2)

@app.get("/api/attendance")
async def get_api_attendance():
    return load_attendance()

@app.get("/attendance")
async def get_attendance():
    return load_attendance()

@app.delete("/attendance/{record_id}")
def delete_attendance_record(record_id: str):
    records = load_attendance()
    record = next((r for r in records if r["id"] == record_id), None)
    
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found")
        
    # Delete the stored image using the URL path if applicable
    img_url = record.get("image")
    if img_url and "/api/images/" in img_url:
        filename = img_url.split("/")[-1]
        img_path = os.path.join(DATA_DIR, "sync_images", filename)
        if os.path.exists(img_path):
            os.remove(img_path)
            
    # Filter out and save
    records = [r for r in records if r["id"] != record_id]
    save_attendance(records)
    
    return {"success": True}

@app.get("/api/images/{filename}")
def get_image(filename: str):
    path = os.path.join(DATA_DIR, "sync_images", filename)
    if os.path.exists(path):
        resp = FileResponse(path)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/api/faces/{filename}")
def get_face(filename: str):
    path = os.path.join(FACES_DIR, filename)
    if os.path.exists(path):
        resp = FileResponse(path)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    raise HTTPException(status_code=404, detail="Face image not found")

@app.post("/sync_offline_data")
@app.post("/api/sync/upload")
def sync_offline_data(files: List[UploadFile] = File(...), timestamps: Optional[str] = Form(None)):
    """
    Receive batch of images from mobile app, recognize faces, and record attendance
    timestamps: JSON string mapping filename -> timestamp
    """
    import shutil
    from datetime import datetime
    
    upload_dir = os.path.join(DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    students = load_students()
    attendance_records = load_attendance()
    processed_count = 0
    success_count = 0
    
    # Parse timestamps map if provided
    ts_map = {}
    if timestamps:
        try:
            ts_map = json.loads(timestamps)
        except:
            pass

    results = []

    for file in files:
        try:
            # 1. Save temp file
            temp_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 2. Process with DeepFace using find() for performance
            # DeepFace.find uses vector embeddings which is much faster than 1:N verify loop
            best_match = None
            
            try:
                # find returns a list of DataFrames (one per face in image)
                dfs = DeepFace.find(
                    img_path=temp_path,
                    db_path=FACES_DIR,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False,
                    silent=True
                )
                
                # Check if any match found in any detected face
                for df in dfs:
                    if not df.empty:
                        # Get best match (first row is usually best distance)
                        match_path = df.iloc[0]['identity']
                        # match_path matches the db structure: student_data/faces/ID.jpg
                        # We need to extract ID. 
                        # identity col contains absolute or relative path depending on version, 
                        # but normally it preserves structure if db_path was relative.
                        # Let's handle filename extraction robustly
                        filename = os.path.basename(match_path)
                        student_id = os.path.splitext(filename)[0]
                        
                        # Find student object
                        best_match = next((s for s in students if s["id"] == student_id), None)
                        if best_match:
                            break
            except Exception as e:
                print(f"DeepFace.find error: {e}")
                
            # 3. Record Attendance
            name = best_match["name"] if best_match else "Unknown Worker"
            student_id = best_match["id"] if best_match else "unknown"
            
            record_ts = ts_map.get(file.filename, datetime.now().isoformat())
            
            # Save the image for dashboard thumbnail
            perm_dir = os.path.join(DATA_DIR, "sync_images")
            os.makedirs(perm_dir, exist_ok=True)
            new_filename = f"{uuid.uuid4()}_{file.filename}"
            perm_path = os.path.join(perm_dir, new_filename)
            shutil.move(temp_path, perm_path)
            
            image_url = f"/api/images/{new_filename}"
            
            record = {
                "id": str(uuid.uuid4()),
                "student_id": student_id,
                "student_name": name,
                "person_name": name,
                "name": name,
                "timestamp": record_ts,
                "image": image_url,
                "status": "Present",
                "method": "sync",
                "synced_at": datetime.now().isoformat()
            }
            attendance_records.append(record)
            
            if best_match:
                success_count += 1
                results.append({"file": file.filename, "status": "matched", "student": name})
            else:
                results.append({"file": file.filename, "status": "unknown"})
            
            # Cleanup temp if it somehow still exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({"file": file.filename, "status": "error", "error": str(e)})

    save_attendance(attendance_records)
    
    return {
        "processed": processed_count,
        "matched": success_count,
        "details": results
    }


# Dahua Camera Configuration
CAMERA_IP = os.getenv("CAMERA_IP", "")
CAMERA_USER = os.getenv("CAMERA_USER", "")
CAMERA_PASS = os.getenv("CAMERA_PASS", "")

# List of RTSP URL formats to try for Dahua cameras
RTSP_URLS = [
    # Standard Dahua RTSP format (confirmed working)
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1",
    # Alternative paths
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/h264/ch1/main/av_stream",
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/live",
]


@app.get("/api/video_feed")
async def video_feed():
    """
    Stream video from Dahua Camera via RTSP
    """
    from fastapi.responses import StreamingResponse
    import time
    
    def generate_frames():
        cap = None
        working_url = None
        
        # Try each RTSP URL format
        for url in RTSP_URLS:
            # Hide password in log
            safe_url = url.replace(CAMERA_PASS, "****")
            print(f"Trying RTSP: {safe_url}")
            
            test_cap = cv2.VideoCapture(url)
            if test_cap.isOpened():
                # Set resolution to HD (1920x1080)
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                
                # Try to read a frame to confirm it works
                ret, _ = test_cap.read()
                if ret:
                    print(f"SUCCESS! Connected to: {safe_url}")
                    # Get actual resolution
                    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Stream resolution: {width}x{height}")
                    cap = test_cap
                    working_url = url
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if cap is None:
            print("All RTSP URLs failed, falling back to webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Webcam also failed to open")
                return

        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                frame_count += 1
                if frame_count > 30:
                    cap.release()
                    time.sleep(2)
                    print("Reconnecting...")
                    if working_url:
                        cap = cv2.VideoCapture(working_url)
                    else:
                        cap = cv2.VideoCapture(0)
                    frame_count = 0
                continue
            
            frame_count = 0

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    # Port 10000 for Render deployment
    uvicorn.run(app, host="0.0.0.0", port=10000)



