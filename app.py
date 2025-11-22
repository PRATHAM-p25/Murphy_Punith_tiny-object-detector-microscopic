# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time
import requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime
import bcrypt

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo — Auth + MongoDB")

# -------------------------
# Config / Constants
# -------------------------
MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""         # optional: provide drive file id to download model at start
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# -------------------------
# Helper: read Mongo URI from secrets or env
# -------------------------
def get_mongo_uri():
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# -------------------------
# Download helper (optional)
# -------------------------
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

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# -------------------------
# Text size helper (robust across PIL versions)
# -------------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

# -------------------------
# Drawing predictions
# -------------------------
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

# -------------------------
# MongoDB init
# -------------------------
client = None
db = None
fs = None
collection = None
db_error_msg = None

if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Trigger server selection / auth early
        client.server_info()
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
        users_collection = db["users"]
    except errors.OperationFailure:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges.")
        client = None
    except errors.ServerSelectionTimeoutError:
        db_error_msg = ("Could not connect to MongoDB Atlas. Possibly IP whitelist issue.")
        client = None
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"
        client = None
else:
    users_collection = None
    db_error_msg = "Mongo URI not provided (set STREAMLIT secret or MONGO_URI env var)."

# -------------------------
# Auth helpers (bcrypt)
# -------------------------
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed)
    except Exception:
        return False

def create_user(username: str, password: str):
    if not USE_DB or client is None:
        raise RuntimeError("DB not available for user creation.")
    if users_collection.find_one({"username": username}):
        raise ValueError("Username already exists.")
    pw_hash = hash_password(password)
    doc = {"username": username, "password_hash": pw_hash, "created_at": datetime.utcnow()}
    users_collection.insert_one(doc)
    return True

def authenticate_user(username: str, password: str):
    if not USE_DB or client is None:
        raise RuntimeError("DB not available for authentication.")
    doc = users_collection.find_one({"username": username})
    if not doc:
        return False
    stored = doc.get("password_hash")
    if isinstance(stored, str):
        # if stored as string (unexpected), convert
        stored = stored.encode("utf-8")
    return check_password(password, stored)

# -------------------------
# Session state: track logged in user
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -------------------------
# Optional: download model from Drive then load
# -------------------------
if GDRIVE_FILE_ID:
    try:
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
    except Exception as e:
        st.error(f"Failed downloading model from Drive: {e}")

# Load model
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

# -------------------------
# UI: Auth (Sign up / Sign in) in sidebar
# -------------------------
st.sidebar.header("Account")
if st.session_state.user:
    st.sidebar.success(f"Signed in as: {st.session_state.user}")
    if st.sidebar.button("Sign out"):
        st.session_state.user = None
        st.sidebar.info("Signed out.")
else:
    auth_tab = st.sidebar.radio("Choose", ("Sign in", "Sign up"))
    if auth_tab == "Sign up":
        st.sidebar.subheader("Create account")
        new_user = st.sidebar.text_input("Username", key="su_user")
        new_pw = st.sidebar.text_input("Password", type="password", key="su_pw")
        if st.sidebar.button("Create account"):
            if not new_user or not new_pw:
                st.sidebar.error("Provide username and password.")
            elif not USE_DB or client is None:
                st.sidebar.error("DB not configured. Cannot create account.")
            else:
                try:
                    create_user(new_user, new_pw)
                    st.sidebar.success("Account created — you can now sign in.")
                except ValueError as ve:
                    st.sidebar.error(str(ve))
                except Exception as e:
                    st.sidebar.error(f"Failed to create account: {e}")
    else:
        st.sidebar.subheader("Sign in")
        in_user = st.sidebar.text_input("Username", key="si_user")
        in_pw = st.sidebar.text_input("Password", type="password", key="si_pw")
        if st.sidebar.button("Sign in"):
            if not in_user or not in_pw:
                st.sidebar.error("Provide username and password.")
            elif not USE_DB or client is None:
                st.sidebar.error("DB not configured. Cannot sign in.")
            else:
                try:
                    ok = authenticate_user(in_user, in_pw)
                    if ok:
                        st.session_state.user = in_user
                        st.sidebar.success("Signed in.")
                    else:
                        st.sidebar.error("Invalid username or password.")
                except Exception as e:
                    st.sidebar.error(f"Auth failed: {e}")

# If not signed in, block main detection UI
if not st.session_state.user:
    st.title("Microscopy Detector — sign in to continue")
    if db_error_msg:
        st.warning(db_error_msg)
    st.info("Create an account or sign in from the left sidebar. Accounts are saved to your MongoDB Atlas.")
    st.stop()

# -------------------------
# Main detection UI (user is signed in)
# -------------------------
st.header("Run detection (signed-in users only)")
col1, col2 = st.columns([1, 1.2])

with col1:
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

            # Save detection to DB if available
            if not USE_DB or client is None:
                st.info("DB not configured. Skipping DB save.")
            else:
                if db_error_msg:
                    st.error(db_error_msg)
                else:
                    try:
                        buf = io.BytesIO()
                        pil_out.save(buf, format="PNG")
                        img_bytes_out = buf.getvalue()
                        # Save image bytes to GridFS
                        file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")
                        document = {
                            "timestamp": datetime.utcnow(),
                            "counts": counts,
                            "model": MODEL_LOCAL_PATH,
                            "username": st.session_state.user,
                            "img_gridfs_id": file_id,
                        }
                        insertion_result = collection.insert_one(document)
                        st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                    except Exception as e:
                        st.error(f"Failed to save to DB: {e}")

with col2:
    st.markdown("### Quick account info")
    st.write(f"Signed in as: **{st.session_state.user}**")
    st.write("Model:", MODEL_LOCAL_PATH)
    if db_error_msg:
        st.error(db_error_msg)
    else:
        st.write("MongoDB: connected")

# End of file
