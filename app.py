# app.py
"""
Microscopy Detector Streamlit App with user sign-up / sign-in and MongoDB Atlas storage.

Features:
- Sign up / Sign in (username + password hashed with bcrypt)
- Save detection images + counts to MongoDB (GridFS + metadata) under the signed-in user
- ONNX model inference via ultralytics.YOLO (uses best.onnx exported from Colab)
- Safe retrieval of MongoDB URI from Streamlit secrets or environment variable MONGO_URI
- Small, clear UI and error handling
"""

import os
import io
import time
import base64
import requests
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# model and db libs
from ultralytics import YOLO
from pymongo import MongoClient, errors
import gridfs
import bcrypt

# -----------------------------
# Configuration / constants
# -----------------------------
MODEL_LOCAL_PATH = "best.onnx"   # place best.onnx in repo or set GDRIVE_FILE_ID and it'll download
GDRIVE_FILE_ID = ""              # optional: Google Drive file id to download model at startup
MODEL_IMG_SIZE = 1024            # model input size you exported with
DEFAULT_CONF = 0.25

# -----------------------------
# Helpers: Mongo URI retrieval
# -----------------------------
def get_mongo_uri():
    # 1) try streamlit secrets (works on Streamlit Cloud)
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

# -----------------------------
# Helpers: GDrive download
# -----------------------------
def download_from_gdrive(file_id, dest):
    """Download a small file from google drive public link (by file id)."""
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

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# -----------------------------
# Text size helper (cross-platform)
# -----------------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

# -----------------------------
# Draw predictions function
# -----------------------------
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
            # score
            try:
                score = float(box.conf[0]) if hasattr(box, "conf") else float(box.confidence)
            except Exception:
                score = float(getattr(box, "confidence", 0.0))
            # class
            try:
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            except Exception:
                cls = int(getattr(box, "class_id", 0))
            if score < conf_thresh:
                continue
            # coords
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

# -----------------------------
# MongoDB connection and helpers
# -----------------------------
client = None
db = None
fs = None
collection = None
users_col = None
db_error_msg = None

if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # trigger server selection / auth errors early
        client.server_info()
        db = client.get_database("microscopy_db")
        fs = gridfs.GridFS(db)
        collection = db.get_collection("detections")
        users_col = db.get_collection("users")
    except errors.OperationFailure as e:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges. "
                        "Ensure user has write access to the specified DB.")
    except errors.ServerSelectionTimeoutError as e:
        db_error_msg = ("Could not connect to MongoDB Atlas. Possibly IP access list issue. "
                        "For testing add 0.0.0.0/0 to Network Access (temporarily).")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# -----------------------------
# Streamlit UI and auth logic
# -----------------------------
st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo (Auth + DB)")
st.title("Microscopy Detector (ONNX + MongoDB)")

# If model is hosted on Drive, download it
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Downloading model failed: {e}")

# Load model
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Authentication UI (left) and Detection UI (right)
col_auth, col_action = st.columns([1, 2])

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None  # stores username when logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "message" not in st.session_state:
    st.session_state.message = ""

with col_auth:
    st.header("Account")
    if st.session_state.logged_in:
        st.success(f"Signed in as: {st.session_state.user}")
        if st.button("Sign out"):
            st.session_state.user = None
            st.session_state.logged_in = False
            st.session_state.message = "Signed out."
            st.experimental_rerun()
        st.write("---")
        st.write("You can save detections to your account. Saved results are private to your user.")
    else:
        # Tabs for sign in / sign up
        tab1, tab2 = st.tabs(["Sign in", "Sign up"])

        with tab1:
            st.subheader("Sign in")
            si_username = st.text_input("Username (sign in)", key="si_user")
            si_password = st.text_input("Password", type="password", key="si_pass")
            if st.button("Sign in"):
                if not USE_DB:
                    st.error("DB not configured. Sign-in requires MongoDB URI in Streamlit secrets or MONGO_URI env var.")
                elif db_error_msg:
                    st.error(db_error_msg)
                else:
                    if not si_username or not si_password:
                        st.error("Please provide username and password.")
                    else:
                        try:
                            user_doc = users_col.find_one({"username": si_username})
                            if not user_doc:
                                st.error("User not found. Please sign up first.")
                            else:
                                stored_hash = user_doc.get("password_hash")
                                if stored_hash and bcrypt.checkpw(si_password.encode("utf-8"), stored_hash.encode("utf-8")):
                                    st.session_state.user = si_username
                                    st.session_state.logged_in = True
                                    st.success("Signed in successfully.")
                                else:
                                    st.error("Invalid username/password.")
                        except Exception as e:
                            st.error(f"Sign-in failed: {e}")

        with tab2:
            st.subheader("Sign up")
            su_username = st.text_input("Username (sign up)", key="su_user")
            su_password = st.text_input("Password", type="password", key="su_pass")
            su_password2 = st.text_input("Confirm password", type="password", key="su_pass2")
            if st.button("Create account"):
                if not USE_DB:
                    st.error("DB not configured. Sign-up requires MongoDB URI in Streamlit secrets or MONGO_URI env var.")
                elif db_error_msg:
                    st.error(db_error_msg)
                else:
                    if not su_username or not su_password or not su_password2:
                        st.error("Please fill all fields.")
                    elif su_password != su_password2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            # ensure username uniqueness
                            if users_col.find_one({"username": su_username}):
                                st.error("Username already exists. Choose another one.")
                            else:
                                # hash password
                                salt = bcrypt.gensalt()
                                pw_hash = bcrypt.hashpw(su_password.encode("utf-8"), salt).decode("utf-8")
                                user_doc = {
                                    "username": su_username,
                                    "password_hash": pw_hash,
                                    "created_at": datetime.utcnow()
                                }
                                users_col.insert_one(user_doc)
                                st.success("Account created. You can now sign in.")
                        except Exception as e:
                            st.error(f"Sign-up failed: {e}")

# -----------------------------
# Detection UI: only show main actions regardless of auth,
# but saving to DB requires login.
# -----------------------------
with col_action:
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

            # Offer save option only if DB available and user logged in
            if not USE_DB:
                st.info("MongoDB not configured. To enable saving detections, add MONGO_URI to Streamlit secrets or MONGO_URI env var.")
            elif db_error_msg:
                st.error(db_error_msg)
            else:
                if not st.session_state.logged_in:
                    st.info("Sign in to save this detection to your account.")
                else:
                    if st.button("Save detection to DB"):
                        try:
                            # Save image bytes into GridFS and metadata into collection
                            buf = io.BytesIO()
                            pil_out.save(buf, format="PNG")
                            img_bytes_out = buf.getvalue()

                            file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")
                            document = {
                                "timestamp": datetime.utcnow(),
                                "counts": counts,
                                "model": MODEL_LOCAL_PATH,
                                "img_gridfs_id": file_id,
                                "saved_by": st.session_state.user,
                            }
                            insertion_result = collection.insert_one(document)
                            st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                        except Exception as e:
                            st.error(f"Failed to save to DB: {e}")

# -----------------------------
# Footer: show helpful tips & DB status
# -----------------------------
st.markdown("---")
if USE_DB:
    if db_error_msg:
        st.error(f"DB: {db_error_msg}")
    else:
        st.write("DB: Connected to MongoDB Atlas (saving enabled).")
else:
    st.info("DB: Not configured. Add your MongoDB URI in Streamlit secrets under `mongo.uri` or set MONGO_URI env var.")

st.caption("Note: passwords are stored hashed (bcrypt). For deployment, never hardcode credentials in code.")
