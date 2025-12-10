# app.py
"""
Streamlit Image Toolkit (fixed)
Features:
 - Background image (local path used)
 - Upload image, apply filters, transformations
 - Feature extraction: LBP, Hu Moments, ORB
 - Undo / Redo history
 - Comparison (split) view with robust channel handling
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import math
from typing import Tuple, Dict, Any, Optional

# ---------- Config ----------
st.set_page_config(page_title="Image Toolkit (Fixed)", layout="wide", initial_sidebar_state="expanded")

# Use the local uploaded file path as background (deployment may transform to URL)
SAMPLE_BG = "/mnt/data/Perception_Playground_Full_Report (AutoRecovered) (AutoRecovered).pdf"
# If you want PNG background instead, replace SAMPLE_BG with:
# SAMPLE_BG = "/mnt/data/955a68f8-71ee-4efa-86ed-09e574d6506e.png"

_page_bg_css = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background-image: url("{SAMPLE_BG}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  opacity: 0.95;
}}
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.22);
  pointer-events: none;
}}
</style>
"""
st.markdown(_page_bg_css, unsafe_allow_html=True)

# ---------- Helpers ----------
HISTORY_LIMIT = 20

def read_image(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return img

def cv2_to_pil(img: np.ndarray) -> Optional[Image.Image]:
    if img is None:
        return None
    if img.ndim == 2:
        return Image.fromarray(img)
    ch = img.shape[2]
    if ch == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if ch == 4:
        # BGRA -> RGBA
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(img)

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr

def to_bytes(img: np.ndarray, fmt: str = 'PNG') -> bytes:
    if img is None:
        return b''
    pil = cv2_to_pil(img)
    buf = io.BytesIO()
    fmt_up = fmt.upper()
    save_fmt = 'JPEG' if fmt_up in ('JPG', 'JPEG') else fmt_up
    pil.save(buf, format=save_fmt)
    return buf.getvalue()

def safe_copy(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if img is None else img.copy()

def ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    return k if (k % 2 == 1) else k+1

def get_image_info(img: Optional[np.ndarray], uploaded_name: Optional[str] = None) -> Dict[str, Any]:
    if img is None:
        return {}
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    return {'Height': h, 'Width': w, 'Channels': c, 'Filename': uploaded_name or 'N/A'}

def ensure_3ch_bgr(img: np.ndarray) -> np.ndarray:
    """
    Convert input image into 3-channel BGR:
      - grayscale -> BGR
      - BGRA/RGBA -> BGR (drops alpha)
      - RGB -> BGR
      - if already BGR (3ch) just return copy
    """
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ch = img.shape[2]
    if ch == 3:
        # We assume it's BGR already (if it's RGB coming from PIL, convert when needed)
        return img.copy()
    if ch == 4:
        # BGRA or RGBA depends on how it was loaded; OpenCV usually gives BGRA for PNG
        # Convert BGRA -> BGR (drops alpha)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # fallback
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGR)  # should not happen but safe

# ---------- Session state init ----------
st.session_state.setdefault('original_img', None)
st.session_state.setdefault('processed_img', None)
st.session_state.setdefault('history', [])  # list of copies
st.session_state.setdefault('redo_stack', [])
st.session_state.setdefault('uploaded_name', None)
st.session_state.setdefault('zoom', 1.0)
st.session_state.setdefault('dark', False)

def push_history(img: Optional[np.ndarray]):
    if img is None:
        return
    # append a copy, trim if needed
    st.session_state.history.append(safe_copy(img))
    if len(st.session_state.history) > HISTORY_LIMIT:
        # drop oldest
        st.session_state.history.pop(0)

def undo():
    if len(st.session_state.history) <= 1:
        return
    current = st.session_state.history.pop()  # remove latest
    st.session_state.redo_stack.append(current)
    st.session_state.processed_img = safe_copy(st.session_state.history[-1])

def redo():
    if not st.session_state.redo_stack:
        return
    state = st.session_state.redo_stack.pop()
    st.session_state.history.append(safe_copy(state))
    st.session_state.processed_img = safe_copy(state)

# ---------- Image ops ----------
def apply_brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    if img is None:
        return None
    b = int(np.clip(brightness, -255, 255))
    c = int(np.clip(contrast, -127, 127))
    out = img.astype(np.int16)
    out = np.clip(out * (1 + c/127.0) + b, 0, 255).astype(np.uint8)
    return out

def rotate90(img: np.ndarray, times: int = 1) -> np.ndarray:
    if img is None:
        return None
    k = times % 4
    return np.ascontiguousarray(np.rot90(img, k=k))

def flip(img: np.ndarray, mode: str = 'h') -> np.ndarray:
    if img is None:
        return None
    if mode == 'h':
        return cv2.flip(img, 1)
    if mode == 'v':
        return cv2.flip(img, 0)
    return img

def crop_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if img is None:
        return None
    H, W = img.shape[:2]
    x1 = int(np.clip(x, 0, W-1))
    y1 = int(np.clip(y, 0, H-1))
    x2 = int(np.clip(x + w, 0, W))
    y2 = int(np.clip(y + h, 0, H))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]

def convert_color(img: np.ndarray, op: str) -> np.ndarray:
    if img is None:
        return None
    if op == 'Grayscale':
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if op == 'BGR -> RGB':
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if op == 'RGB -> BGR':
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if op == 'BGR -> HSV':
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if op == 'BGR -> YCrCb':
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img

def apply_filter(img: np.ndarray, filter_name: str, ksize: int = 3) -> np.ndarray:
    if img is None:
        return None
    k = ensure_odd(max(1, ksize))
    if filter_name == 'Gaussian':
        return cv2.GaussianBlur(img, (k,k), 0)
    if filter_name == 'Mean':
        return cv2.blur(img, (k,k))
    if filter_name == 'Median':
        return cv2.medianBlur(img, k)
    if filter_name == 'Sobel':
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(sx, sy)
        return np.uint8(np.clip(mag, 0, 255))
    if filter_name == 'Laplacian':
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return np.uint8(np.clip(np.abs(lap), 0, 255))
    return img

def morphology(img: np.ndarray, op: str, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    if img is None:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if op == 'Dilation':
        return cv2.dilate(img, kernel, iterations=iterations)
    if op == 'Erosion':
        return cv2.erode(img, kernel, iterations=iterations)
    if op == 'Opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if op == 'Closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def edge_detection(img: np.ndarray, method: str, thresh1: int = 100, thresh2: int = 200) -> np.ndarray:
    if img is None:
        return None
    if method == 'Canny':
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, thresh1, thresh2)
    if method == 'Sobel':
        return apply_filter(img, 'Sobel')
    if method == 'Laplacian':
        return apply_filter(img, 'Laplacian')
    return img

# ---------- Feature extraction ----------
def lbp_image(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    out = np.zeros((H-2, W-2), dtype=np.uint8)
    for i in range(1, H-1):
        top = gray[i-1]
        mid = gray[i]
        bot = gray[i+1]
        for j in range(1, W-1):
            c = int(mid[j])
            code = 0
            code |= (1 << 7) if int(top[j-1]) >= c else 0
            code |= (1 << 6) if int(top[j])   >= c else 0
            code |= (1 << 5) if int(top[j+1]) >= c else 0
            code |= (1 << 4) if int(mid[j+1]) >= c else 0
            code |= (1 << 3) if int(bot[j+1]) >= c else 0
            code |= (1 << 2) if int(bot[j])   >= c else 0
            code |= (1 << 1) if int(bot[j-1]) >= c else 0
            code |= (1 << 0) if int(mid[j-1]) >= c else 0
            out[i-1,j-1] = code
    return out

def hu_moments_log10(gray: np.ndarray) -> list:
    M = cv2.moments(gray)
    hu = cv2.HuMoments(M).flatten()
    hu_log = []
    for h in hu:
        if h == 0:
            hu_log.append(0.0)
        else:
            hu_log.append(-1.0 * math.copysign(1.0, h) * math.log10(abs(h)))
    return hu_log

def orb_keypoints_visual(img: np.ndarray, nfeatures: int = 500) -> Tuple[np.ndarray, int]:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures)
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    vis = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis, len(kp)

def compute_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
    prob = hist / (gray.size + 1e-12)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))

def extract_features(img: np.ndarray, method: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if img is None:
        return None, {}
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = method.lower()
    if method == 'lbp':
        lbp = lbp_image(gray)
        hist = np.bincount(lbp.ravel(), minlength=256).tolist()
        return lbp, {'lbp_histogram_len': len(hist), 'entropy': float(compute_entropy(gray))}
    if method == 'hu moments':
        hu = hu_moments_log10(gray)
        return None, {'hu_log10': hu}
    if method == 'orb keypoints':
        vis, count = orb_keypoints_visual(img, nfeatures=1000)
        return vis, {'orb_keypoint_count': int(count)}
    return None, {}

# ---------- UI layout ----------
col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.markdown("**File**")
with col2:
    st.markdown("### Image Toolkit — Fixed")
with col3:
    if st.button("Exit"):
        st.stop()

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Open → Upload an image", type=['png','jpg','jpeg','bmp','tiff'])

if uploaded_file is not None:
    img = read_image(uploaded_file)
    if img is not None:
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.original_img = safe_copy(img)
        st.session_state.processed_img = safe_copy(img)
        st.session_state.history = [safe_copy(img)]
        st.session_state.redo_stack = []

op_category = st.sidebar.selectbox('Category', [
    'Image Info', 'Color Conversions', 'Transformations', 'Filtering & Morphology',
    'Enhancement', 'Edge Detection', 'Compression & Save', 'Quick Tools', 'Feature Extraction'
])

# ----------------- Controls Implementation -----------------
if op_category == 'Quick Tools':
    st.sidebar.markdown("### Quick Tools")
    if st.sidebar.button("Rotate 90° CW"):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = rotate90(st.session_state.processed_img, times=3)
            push_history(st.session_state.processed_img)
    if st.sidebar.button("Rotate 90° CCW"):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = rotate90(st.session_state.processed_img, times=1)
            push_history(st.session_state.processed_img)
    if st.sidebar.button("Flip Horizontal"):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = flip(st.session_state.processed_img, 'h')
            push_history(st.session_state.processed_img)
    if st.sidebar.button("Flip Vertical"):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = flip(st.session_state.processed_img, 'v')
            push_history(st.session_state.processed_img)
    st.sidebar.markdown("---")
    if st.sidebar.button("Preset: High Contrast"):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = apply_brightness_contrast(st.session_state.processed_img, brightness=0, contrast=80)
            push_history(st.session_state.processed_img)
    if st.sidebar.button("Preset: Soft Blur"):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = apply_filter(st.session_state.processed_img, 'Gaussian', ksize=7)
            push_history(st.session_state.processed_img)
    st.sidebar.markdown("---")
    if st.sidebar.button("Undo"):
        undo()
    if st.sidebar.button("Redo"):
        redo()

elif op_category == 'Color Conversions':
    choice = st.sidebar.selectbox('Convert', ['Grayscale', 'BGR -> RGB', 'RGB -> BGR', 'BGR -> HSV', 'BGR -> YCrCb'])
    if st.sidebar.button('Apply Color Conversion'):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = convert_color(st.session_state.processed_img, choice)
            push_history(st.session_state.processed_img)

elif op_category == 'Transformations':
    trans = st.sidebar.selectbox('Transform', ['Rotate Arbitrary', 'Scale', 'Translate', 'Crop'])
    if trans == 'Rotate Arbitrary':
        angle = st.sidebar.slider('Angle', -180, 180, 0)
        if st.sidebar.button('Apply Rotate'):
            img = st.session_state.processed_img
            if img is not None:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                st.session_state.processed_img = cv2.warpAffine(img, M, (w, h))
                push_history(st.session_state.processed_img)
    elif trans == 'Scale':
        fx = st.sidebar.slider('Scale X (fx)', 0.1, 3.0, 1.0)
        fy = st.sidebar.slider('Scale Y (fy)', 0.1, 3.0, 1.0)
        if st.sidebar.button('Apply Scale'):
            img = st.session_state.processed_img
            if img is not None:
                st.session_state.processed_img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
                push_history(st.session_state.processed_img)
    elif trans == 'Translate':
        tx = st.sidebar.slider('Translate X', -500, 500, 0)
        ty = st.sidebar.slider('Translate Y', -500, 500, 0)
        if st.sidebar.button('Apply Translate'):
            img = st.session_state.processed_img
            if img is not None:
                h, w = img.shape[:2]
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                st.session_state.processed_img = cv2.warpAffine(img, M, (w, h))
                push_history(st.session_state.processed_img)
    elif trans == 'Crop':
        st.sidebar.write("Simple crop (x,y,width,height)")
        x = st.sidebar.number_input('x', min_value=0, value=0, step=1)
        y = st.sidebar.number_input('y', min_value=0, value=0, step=1)
        cw = st.sidebar.number_input('width', min_value=1, value=100, step=1)
        ch = st.sidebar.number_input('height', min_value=1, value=100, step=1)
        if st.sidebar.button('Apply Crop'):
            if st.session_state.processed_img is not None:
                st.session_state.processed_img = crop_image(st.session_state.processed_img, x, y, cw, ch)
                push_history(st.session_state.processed_img)

elif op_category == 'Filtering & Morphology':
    choice = st.sidebar.selectbox('Filter/Morph', ['Gaussian','Mean','Median','Sobel','Laplacian','Dilation','Erosion','Opening','Closing'])
    k = st.sidebar.slider('Kernel size', 1, 31, 3, step=2)
    iters = st.sidebar.slider('Iterations (morphology)', 1, 10, 1)
    if st.sidebar.button('Apply'):
        if st.session_state.processed_img is not None:
            if choice in ['Gaussian','Mean','Median','Sobel','Laplacian']:
                st.session_state.processed_img = apply_filter(st.session_state.processed_img, choice, ksize=k)
            else:
                st.session_state.processed_img = morphology(st.session_state.processed_img, choice, ksize=k, iterations=iters)
            push_history(st.session_state.processed_img)

elif op_category == 'Enhancement':
    choice = st.sidebar.selectbox('Enhance', ['Histogram Equalization','Sharpen','Contrast Stretch','Brightness/Contrast'])
    if choice == 'Brightness/Contrast':
        brightness = st.sidebar.slider('Brightness', -100, 100, 0)
        contrast = st.sidebar.slider('Contrast', -80, 80, 0)
        if st.sidebar.button('Apply'):
            if st.session_state.processed_img is not None:
                st.session_state.processed_img = apply_brightness_contrast(st.session_state.processed_img, brightness, contrast)
                push_history(st.session_state.processed_img)
    else:
        if st.sidebar.button('Apply Enhance'):
            img = st.session_state.processed_img
            if img is not None:
                if choice == 'Histogram Equalization':
                    if img.ndim == 3:
                        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                        st.session_state.processed_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                    else:
                        st.session_state.processed_img = cv2.equalizeHist(img)
                elif choice == 'Sharpen':
                    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                    st.session_state.processed_img = cv2.filter2D(img, -1, kernel)
                elif choice == 'Contrast Stretch':
                    in_min = np.min(img)
                    in_max = np.max(img)
                    out = (img - in_min) * (255.0 / (in_max - in_min + 1e-8))
                    st.session_state.processed_img = np.uint8(out)
                push_history(st.session_state.processed_img)

elif op_category == 'Edge Detection':
    method = st.sidebar.selectbox('Method', ['Canny','Sobel','Laplacian'])
    t1 = st.sidebar.slider('Canny Thresh1', 0, 500, 100)
    t2 = st.sidebar.slider('Canny Thresh2', 0, 500, 200)
    if st.sidebar.button('Apply Edge Detection'):
        if st.session_state.processed_img is not None:
            st.session_state.processed_img = edge_detection(st.session_state.processed_img, method, t1, t2)
            push_history(st.session_state.processed_img)

elif op_category == 'Compression & Save':
    fmt = st.sidebar.selectbox('Save format', ['PNG','JPG','BMP'])
    quality = st.sidebar.slider('JPEG Quality (if JPG)', 10, 100, 90)
    if st.sidebar.button('Save Processed Image'):
        if st.session_state.processed_img is not None:
            b = to_bytes(st.session_state.processed_img, fmt=fmt)
            st.sidebar.download_button('Download', data=b, file_name=f'processed.{fmt.lower()}', mime=f'image/{fmt.lower()}')

# Image Info sidebar
if op_category == 'Image Info':
    st.sidebar.markdown('---')
    st.sidebar.write('Image fundamentals and metadata')
    if st.session_state.processed_img is not None:
        info = get_image_info(st.session_state.processed_img, st.session_state.uploaded_name)
        st.sidebar.write(info)
    else:
        st.sidebar.write("No image loaded.")

# Feature Extraction
if op_category == 'Feature Extraction':
    st.sidebar.markdown("### Feature Extraction")
    feat_choice = st.sidebar.selectbox('Method', ['LBP', 'Hu Moments', 'ORB Keypoints'])
    if st.sidebar.button('Apply Feature Extraction'):
        img = st.session_state.processed_img
        if img is None:
            st.sidebar.warning("No image loaded")
        else:
            vis, feats = extract_features(img, feat_choice)
            if vis is not None:
                # LBP returns single-channel (show as grayscale) -> convert to 3ch for consistency
                if vis.ndim == 2:
                    st.session_state.processed_img = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                else:
                    st.session_state.processed_img = vis
                push_history(st.session_state.processed_img)
            if feats:
                st.sidebar.markdown("**Extracted features**")
                for k, v in feats.items():
                    if isinstance(v, (list, np.ndarray)) and len(v) > 50:
                        st.sidebar.write(f"{k}: array (len={len(v)})")
                    else:
                        st.sidebar.write(f"{k}: {v}")

# ----------------- Display area (robust concat) -----------------
compare = st.sidebar.checkbox('Comparison mode (split view)', value=False)
left_col, right_col = st.columns(2)

with left_col:
    st.subheader('Original Image')
    if st.session_state.original_img is not None:
        # convert to PIL for display; cv2_to_pil handles channel conversions
        st.image(cv2_to_pil(st.session_state.original_img), use_container_width=True)
    else:
        st.info('No image uploaded yet.')

with right_col:
    st.subheader('Processed Image')
    if st.session_state.processed_img is not None:
        if compare and st.session_state.original_img is not None:
            # Ensure both are 3-channel BGR arrays and same dtype
            a = ensure_3ch_bgr(st.session_state.original_img)
            b = ensure_3ch_bgr(st.session_state.processed_img)
            # crop to min height/width
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            left_half = a[:h, :w].copy()
            right_half = b[:h, :w].copy()
            # if either is still single-channel (shouldn't), convert
            if left_half.ndim == 2:
                left_half = cv2.cvtColor(left_half, cv2.COLOR_GRAY2BGR)
            if right_half.ndim == 2:
                right_half = cv2.cvtColor(right_half, cv2.COLOR_GRAY2BGR)
            # ensure both have same number of channels (3)
            if left_half.shape[2] != right_half.shape[2]:
                left_half = ensure_3ch_bgr(left_half)
                right_half = ensure_3ch_bgr(right_half)
            # build split: left image left half + right image right half
            mid = w // 2
            left_piece = left_half[:, :mid]
            right_piece = right_half[:, mid:]
            # If widths are mismatched due to odd widths, adjust
            if left_piece.shape[1] != right_piece.shape[1]:
                # pad the smaller to match width
                target_w = max(left_piece.shape[1], right_piece.shape[1])
                def pad_to_width(img_arr, target_width):
                    h, w0 = img_arr.shape[:2]
                    if w0 >= target_width:
                        return img_arr
                    pad_w = target_width - w0
                    pad = np.zeros((h, pad_w, img_arr.shape[2]), dtype=img_arr.dtype)
                    return np.concatenate([img_arr, pad], axis=1)
                left_piece = pad_to_width(left_piece, target_w)
                right_piece = pad_to_width(right_piece, target_w)
            split = np.concatenate([left_piece, right_piece], axis=1)
            st.image(cv2_to_pil(split), use_container_width=True)
        else:
            st.image(cv2_to_pil(st.session_state.processed_img), use_container_width=True)
    else:
        st.info('No processed image available.')

# Footer / download / reset
st.markdown('---')
if st.session_state.processed_img is not None:
    info = get_image_info(st.session_state.processed_img, st.session_state.uploaded_name)
    size_bytes = len(to_bytes(st.session_state.processed_img, fmt='PNG'))
    st.caption(f"Dimensions: {info.get('Height')} x {info.get('Width')} | Channels: {info.get('Channels')} | File: {info.get('Filename')} | Estimated size (PNG): {size_bytes} bytes")
else:
    st.caption('No image loaded')

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button('Reset to Original') and st.session_state.original_img is not None:
        st.session_state.processed_img = safe_copy(st.session_state.original_img)
        st.session_state.history = [safe_copy(st.session_state.original_img)]
        st.session_state.redo_stack = []
with col_b:
    if st.button('Save as PNG') and st.session_state.processed_img is not None:
        b = to_bytes(st.session_state.processed_img, fmt='PNG')
        st.download_button('Download PNG', data=b, file_name='processed.png', mime='image/png')
with col_c:
    if st.button('Save as JPG') and st.session_state.processed_img is not None:
        b = to_bytes(st.session_state.processed_img, fmt='JPEG')
        st.download_button('Download JPG', data=b, file_name='processed.jpg', mime='image/jpeg')
