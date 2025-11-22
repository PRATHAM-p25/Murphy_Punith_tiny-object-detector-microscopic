import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64
import requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime
import bcrypt
from bson.objectid import ObjectId

# -----------------------
# Config
# -----------------------
st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo (Auth + Mongo)")

# Model / DB settings - change if needed
MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""          # optional: set if you want app to download model from Drive
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# -----------------------
# Helpers: Mongo URI, password hashing
# -----------------------
def get_mongo_uri():
    # Prefer Streamlit secrets, fallback to env var
    try:
        # Check for st.secrets existence and structure
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        # st.secrets might not be defined or accessible
        pass
    # Fallback to environment variable
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

def hash_password(plain_password: str) -> bytes:
    """Hashes a plain text password using bcrypt."""
    # The stored password needs to be checked against the salt + hash
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())

def check_password(plain_password: str, hashed: bytes) -> bool:
    """Checks a plain text password against the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed)
    except Exception:
        return False

# -----------------------
# Download model helper
# -----------------------
def download_from_gdrive(file_id, dest):
    """Downloads a file from Google Drive."""
    if os.path.exists(dest):
        return dest
    # Basic download link for public files
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
    """Loads the YOLO model (cached)."""
    return YOLO(model_path)

# -----------------------
# Text size utility (robust across environments)
# -----------------------
def get_text_size(draw, text, font):
    """Calculates text dimensions robustly across Pillow versions."""
    try:
        # Use textbbox (modern Pillow)
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        # Fallback for older versions (textsize is deprecated)
        try:
            return draw.textsize(text, font=font)
        except Exception:
            # Final fallback, non-accurate estimation
            return (len(text) * 6, 11)

# -----------------------
# Draw detections
# -----------------------
def draw_predictions(pil_img, results, conf_thresh=0.25, model_names=None):
    """Draws bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(pil_img)
    try:
        # Attempt to load a common font
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        # Fallback to default font
        font = ImageFont.load_default()
        
    counts = {}
    
    # Iterate through all detection results
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
            
        # Iterate through individual detected bounding boxes
        for box in boxes:
            # Robustly get confidence score
            score = float(box.conf[0]) if hasattr(box, "conf") else float(getattr(box, "confidence", 0.0))
            
            # Robustly get class ID
            cls = int(box.cls[0]) if hasattr(box, "cls") else int(getattr(box, "class_id", 0))
            
            if score < conf_thresh:
                continue
                
            # Robustly get bounding box coordinates (x1, y1, x2, y2)
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
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            
            # Draw label background and text
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            # Position label box slightly above the bounding box
            ty1 = max(0, y1 - th) 
            draw.rectangle([x1, ty1, x1 + tw, y1], fill=(255,0,0))
            draw.text((x1, ty1), text, fill=(255,255,255), font=font)
            
    return pil_img, counts

# -----------------------
# MongoDB init
# -----------------------
client = None
db = None
fs = None
collection = None
users_col = None
db_error_msg = None

if USE_DB:
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Triggers connection/auth check
        
        # Select database and collections
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
        users_col = db["users"]
        
    except errors.OperationFailure:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges.")
    except errors.ServerSelectionTimeoutError:
        db_error_msg = ("Could not connect to MongoDB Atlas. Check Network Access / IP whitelist.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# -----------------------
# UI: Auth (Signin / Signup) at top
# -----------------------
st.markdown("<h1 style='text-align:left'>Microscopy Detector (ONNX via Ultralytics + MongoDB)</h1>", unsafe_allow_html=True)
st.write("---")

# Initialize session state for user
if "user" not in st.session_state:
    st.session_state.user = None  # Stores {'username':..., '_id':...}

# ----------------------------------------------------
# AUTHENTICATION BLOCK (Forced Visibility Container)
# ----------------------------------------------------
# Use a container with a border to isolate the auth section and force rendering
auth_container = st.container(border=True) 

with auth_container:
    st.subheader("User Authentication")
    
    # Check if DB is configured and working
    if not USE_DB or db_error_msg:
        st.error("Authentication Disabled: Cannot connect to MongoDB. Please check MONGO_URI and connection status.")
        if db_error_msg:
             st.warning(f"Connection error: {db_error_msg}")
        else:
             st.warning("MONGO_URI not configured.")

    elif st.session_state.user:
        # User is logged in: show status and Logout button
        st.success(f"Signed in as: **{st.session_state.user.get('username')}**")
        if st.button("Logout", key="logout_btn", use_container_width=True):
            st.session_state.user = None
            st.experimental_rerun()
            
    else:
        # User is NOT logged in: use tabs
        tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])

        # --- Display Sign In Form ---
        with tab_signin:
            st.info("Sign in to save your detections.")
            with st.form("signin_form"):
                si_username = st.text_input("Username or Email", key="si_username")
                si_password = st.text_input("Password", type="password", key="si_password")
                submitted = st.form_submit_button("Sign in", type="primary")
                
                if submitted:
                    if db_error_msg:
                        st.error(db_error_msg)
                    else:
                        # Find user by username or email
                        user = users_col.find_one({"$or": [{"username": si_username}, {"email": si_username}]})
                        if not user:
                            st.error("User not found.")
                        else:
                            stored_pw = user.get("password")
                            # stored_pw may be bytes or Binary in Mongo; handle both
                            if isinstance(stored_pw, (bytes, bytearray)):
                                good = check_password(si_password, stored_pw)
                            else:
                                # Try converting from BSON Binary type if necessary
                                try:
                                    good = check_password(si_password, bytes(stored_pw))
                                except Exception:
                                    good = False
                                    
                            if good:
                                # Store essential user data in session state
                                st.session_state.user = {"username": user.get("username"), "_id": str(user.get("_id"))}
                                st.success(f"Signed in: {user.get('username')}")
                                st.experimental_rerun()
                            else:
                                st.error("Incorrect password.")

        # --- Display Sign Up Form ---
        with tab_signup:
            st.info("Create a new account.")
            with st.form("signup_form"):
                su_username = st.text_input("Username", key="su_username")
                su_email = st.text_input("Email (Optional)", key="su_email")
                su_password = st.text_input("Password", type="password", key="su_password")
                su_password2 = st.text_input("Confirm password", type="password", key="su_password2")
                submitted = st.form_submit_button("Create account", type="primary")
                
                if submitted:
                    if db_error_msg:
                        st.error(db_error_msg)
                    elif not su_username or not su_password:
                        st.error("Provide a username and password.")
                    elif su_password != su_password2:
                        st.error("Passwords do not match.")
                    else:
                        # Check existing user
                        existing = users_col.find_one({"$or": [{"username": su_username}, {"email": su_email}]})
                        if existing:
                            st.error("User with that username or email already exists.")
                        else:
                            hashed = hash_password(su_password)
                            user_doc = {
                                "username": su_username,
                                "email": su_email,
                                "password": hashed,    # bytes
                                "created_at": datetime.utcnow()
                            }
                            try:
                                users_col.insert_one(user_doc)
                                # Successfully created account, prompt user to sign in
                                st.success("Account created. Please switch to the **Sign In** tab to log in.")
                            except Exception as e:
                                st.error(f"Failed to create account: {e}")

# ----------------------------------------------------
# DB Status Block (Reduced Clutter)
# ----------------------------------------------------
with st.expander("Show Model and Database Status"):
    col_model_status, col_db_status = st.columns(2)
    with col_model_status:
        model_status = "Loaded" if os.path.exists(MODEL_LOCAL_PATH) else "Not found locally"
        st.write(f"Model file: `{MODEL_LOCAL_PATH}` — **{model_status}**")
    with col_db_status:
        if USE_DB:
            if db_error_msg:
                st.error(f"DB: Error - {db_error_msg}")
            else:
                st.success("DB: Connected")
        else:
            st.info("DB: Not configured. Add MONGO URI in Streamlit secrets or env var.")

st.write("---")

# -----------------------
# Load model; allow auto-download if Drive ID present
# -----------------------
if GDRIVE_FILE_ID:
    with st.spinner(f"Checking / Downloading model from Drive (ID: {GDRIVE_FILE_ID})..."):
        try:
            download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        except Exception as e:
            st.error(f"Failed to download model from Drive: {e}")

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# -----------------------
# Main interface: detection
# -----------------------
col1, col2 = st.columns([1, 1.2])
with col1:
    st.header("Run Detection")
    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
    uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
    camera = st.camera_input("Or take a picture (Chromium browsers)")

    if uploaded is None and camera is None:
        st.info("Upload an image or use the camera to start.")
    else:
        # Read the image bytes
        img_bytes = uploaded.read() if uploaded else camera.read()
        # Open and convert to RGB (YOLO prefers this)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(pil_img, caption="Input image", width=400)

        if st.button("Run inference", use_container_width=True, type="primary"):
            start = time.time()
            with st.spinner("Processing image and running YOLO model..."):
                try:
                    # Run detection
                    results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf, verbose=False)
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    st.stop()

                # Draw predictions and get counts
                pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf, model_names=model_names)
            
            st.image(pil_out, caption="Detections (Saved to DB if signed in)", use_column_width=True)
            st.write("Counts:", counts)
            st.success(f"Inference done in {time.time()-start:.2f}s")

            # Only save to DB if user signed in
            if not USE_DB:
                st.info("Mongo URI not provided. Skipping DB save.")
            elif db_error_msg:
                st.error(db_error_msg)
            elif not st.session_state.user:
                st.info("Sign in to save this detection to the DB.")
            else:
                try:
                    # 1. Save processed image bytes to GridFS
                    buf = io.BytesIO()
                    # Using JPEG for potentially smaller file size, though PNG is fine too
                    pil_out.save(buf, format="JPEG", quality=90) 
                    img_bytes_out = buf.getvalue()
                    file_id = fs.put(
                        img_bytes_out, 
                        filename=f"det_{st.session_state.user.get('username')}_{int(time.time())}.jpg", 
                        contentType="image/jpeg",
                        user_id=st.session_state.user.get("_id") # Optional metadata
                    )

                    # 2. Save detection metadata to 'detections' collection
                    document = {
                        "timestamp": datetime.utcnow(),
                        "counts": counts,
                        "conf_threshold": conf,
                        "model": MODEL_LOCAL_PATH,
                        "img_gridfs_id": file_id,
                        "user_id": ObjectId(st.session_state.user.get("_id")), # Store as ObjectId
                        "username": st.session_state.user.get("username")
                    }
                    insertion_result = collection.insert_one(document)
                    st.success(f"Saved detection to DB! Document ID: {insertion_result.inserted_id}")
                except Exception as e:
                    st.error(f"Failed to save to DB: {e}")

with col2:
    st.header("Instructions / Quick help")
    st.markdown("""
    This application combines a microscopic object detection model (YOLO ONNX) with user authentication and data persistence using MongoDB Atlas.

    - The Sign In/Sign Up forms are located in the **User Authentication** box at the top of the page.
    - Use the **Sign Up** tab to create an account. Passwords are securely hashed with **bcrypt** before storage.
    - Use the **Sign In** tab to log in. Once logged in, every detection you run will be saved in your history in MongoDB.
    
    ### MongoDB Configuration:
    
    If using Streamlit Cloud, add your MongoDB URI in *Manage app → Settings → Secrets* as:
    
    ```
    # .streamlit/secrets.toml
    [mongo]
    uri = "mongodb+srv://<username>:<password>@cluster0.dkc9xzx.mongodb.net/?retryWrites=true&w=majority"
    ```
    
    Alternatively, set the `MONGO_URI` as an environment variable.
    """)
    
    # Show the current user object for debugging/reference
    st.subheader("Session State")
    st.json(st.session_state.user if st.session_state.user else {"user": "None (Logged Out)"})
