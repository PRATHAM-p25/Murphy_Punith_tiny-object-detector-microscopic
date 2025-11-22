# app.py
# Requirements (add to requirements.txt):
# streamlit
# ultralytics
# opencv-python-headless
# Pillow
# numpy
# requests
# pymongo
# gridfs
# passlib[bcrypt]

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64, requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime
from passlib.hash import bcrypt

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo (Auth + Mongo)")

st.title("Microscopy Detector (ONNX via Ultralytics + MongoDB storage)")

# -----------------------
# Settings (edit as needed)
# -----------------------
MODEL_LOCAL_PATH = "best.onnx"   # path inside the repo / deployed app
GDRIVE_FILE_ID = ""              # optional: Google Drive file id to download model at startup
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# -----------------------
# Helpers: Mongo URI
# -----------------------
def get_mongo_uri():
    # 1) Try Streamlit secrets: [mongo] uri = "..."
    try:
        conf = st.secrets.get("mongo")
        if conf and "uri" in conf:
            return conf["uri"]
    except Exception:
        pass
    # 2) fallback to environment variable
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# -----------------------
# Download helper (optional)
# -----------------------
def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# -----------------------
# Text size helper (robust)
# -----------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

# -----------------------
# Drawing results
# -----------------------
def draw_predictions(pil_img, results, conf_thresh=0.25, model_names=None):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                score = float(box.conf[0]) if hasattr(box, "conf") else float(box.confidence)
            except Exception:
                score = float(getattr(box, "confidence", 0.0))
            try:
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            except Exception:
                cls = int(getattr(box, "class_id", 0))
            if score < conf_thresh:
                continue
            try:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
            except Exception:
                coords = getattr(box, "xyxy", None)
                if coords is not None:
                    x1, y1, x2, y2 = coords[0].tolist()
                else:
                    continue
            label = (model_names[cls] if model_names and cls < len(model_names) else str(cls))
            counts[label] = counts.get(label, 0) + 1
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            ty1 = max(0, y1 - th)
            draw.rectangle([x1, ty1, x1 + tw, y1], fill=(255,0,0))
            draw.text((x1, ty1), text, fill=(255,255,255), font=font)
    return pil_img, counts

# -----------------------
# Mongo connection + GridFS
# -----------------------
client = None
db = None
fs = None
collection = None
db_error_msg = None

if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # raise if cannot connect/auth
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
        users_coll = db["users"]
    except errors.OperationFailure:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges. "
                        "Ensure the user has write rights to the DB/collection.")
    except errors.ServerSelectionTimeoutError:
        db_error_msg = ("Could not connect to MongoDB Atlas. This often means your IP/network is blocked. "
                        "For testing, temporarily allow 0.0.0.0/0 in Atlas Network Access or add your host.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"
else:
    users_coll = None

# -----------------------
# Auth helpers (users collection)
# -----------------------
def create_user(username: str, password: str):
    if not USE_DB or users_coll is None:
        return False, "DB not configured."
    if users_coll.find_one({"username": username}):
        return False, "User already exists."
    pw_hash = bcrypt.hash(password)
    users_coll.insert_one({
        "username": username,
        "password_hash": pw_hash,
        "created_at": datetime.utcnow()
    })
    return True, "User created."

def verify_user(username: str, password: str):
    if not USE_DB or users_coll is None:
        return False, "DB not configured."
    doc = users_coll.find_one({"username": username})
    if not doc:
        return False, "User not found."
    try:
        if bcrypt.verify(password, doc["password_hash"]):
            return True, "OK"
        else:
            return False, "Invalid password."
    except Exception as e:
        return False, f"Verification error: {e}"

# -----------------------
# Session state for auth
# -----------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

# -----------------------
# Optional: download model from Drive at startup
# -----------------------
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Downloading model failed: {e}")

# -----------------------
# Load model
# -----------------------
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# -----------------------
# Top-right: Auth UI
# -----------------------
auth_col1, auth_col2 = st.columns([1, 1])
with auth_col2:
    st.markdown("### Account")
    if st.session_state["user"]:
        st.write(f"Signed in as **{st.session_state['user']}**")
        if st.button("Sign out"):
            st.session_state["user"] = None
            st.success("Signed out.")
    else:
        tabs = st.tabs(["Sign in", "Sign up"])
        with tabs[0]:
            with st.form("signin_form"):
                sin_user = st.text_input("Username", key="sin_user")
                sin_pw = st.text_input("Password", type="password", key="sin_pw")
                submitted = st.form_submit_button("Sign in")
                if submitted:
                    if not USE_DB:
                        st.error("DB not configured. Set MONGO_URI in Streamlit secrets or env.")
                    elif db_error_msg:
                        st.error(db_error_msg)
                    else:
                        ok, msg = verify_user(sin_user.strip(), sin_pw)
                        if ok:
                            st.session_state["user"] = sin_user.strip()
                            st.success("Signed in.")
                        else:
                            st.error(msg)
        with tabs[1]:
            with st.form("signup_form"):
                sup_user = st.text_input("Choose username", key="sup_user")
                sup_pw = st.text_input("Choose password", type="password", key="sup_pw")
                sup_pw2 = st.text_input("Repeat password", type="password", key="sup_pw2")
                submitted2 = st.form_submit_button("Sign up")
                if submitted2:
                    if sup_pw != sup_pw2:
                        st.error("Passwords do not match.")
                    elif not sup_user or not sup_pw:
                        st.error("Provide username and password.")
                    elif not USE_DB:
                        st.error("DB not configured. Set MONGO_URI in Streamlit secrets or env.")
                    elif db_error_msg:
                        st.error(db_error_msg)
                    else:
                        ok, msg = create_user(sup_user.strip(), sup_pw)
                        if ok:
                            st.success("User created. You can sign in now.")
                        else:
                            st.error(msg)

# -----------------------
# Main layout: detection and (no saved list)
# -----------------------
col1, col2 = st.columns([1, 1.2])
with col1:
    st.header("Run detection")
    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
    uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
    camera = st.camera_input("Or take a picture (Chromium browsers)")

    if uploaded is None and camera is None:
        st.info("Upload an image or use the camera.")
    else:
        img_bytes = uploaded.read() if uploaded else camera.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(pil_img, caption="Input image", width=400)

        if st.button("Run inference"):
            start = time.time()
            try:
                results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf, verbose=False)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                st.stop()

            pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf, model_names=model_names)
            st.image(pil_out, caption="Detections", use_column_width=True)
            st.write("Counts:", counts)
            st.success(f"Inference done in {time.time()-start:.2f}s")

            # Save to DB (GridFS + doc) if configured
            if not USE_DB:
                st.info("Mongo URI not provided. Skipping DB save.")
            else:
                if db_error_msg:
                    st.error(db_error_msg)
                else:
                    try:
                        buf = io.BytesIO()
                        pil_out.save(buf, format="PNG")
                        img_bytes_out = buf.getvalue()

                        file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")

                        document = {
                            "timestamp": datetime.utcnow(),
                            "counts": counts,
                            "model": MODEL_LOCAL_PATH,
                            "img_gridfs_id": file_id,
                            "user": st.session_state["user"],   # None if not signed in
                        }
                        insertion_result = collection.insert_one(document)
                        st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                    except Exception as e:
                        st.error(f"Failed to save to DB: {e}")

with col2:
    st.header("Instructions / Info")
    st.markdown(
        """
        * Use the **Sign up** tab to create an account (stores username + hashed password in Atlas).
        * Use the **Sign in** tab to sign in — signed-in username will be attached to saved detection documents.
        * To enable DB storage, set your MongoDB Atlas URI in **Streamlit secrets** as:
        
        ```toml
        [mongo]
        uri = "YOUR_MONGODB_ATLAS_URI"
        ```
        
        or set environment variable `MONGO_URI` before running locally.
        
        * If you deploy to Streamlit Cloud, add the secret via the app's "Settings → Secrets" UI (or set MONGO_URI in the environment).
        * Ensure your Atlas user has write privileges and Network Access allows the host IP (or 0.0.0.0/0 for testing).
        """
    )

# End of file
