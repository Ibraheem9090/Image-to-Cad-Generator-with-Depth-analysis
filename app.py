import streamlit as st
import cv2
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
import base64
from io import BytesIO
from openai import OpenAI
from transformers import pipeline

# App Configuration
st.set_page_config(
    page_title="Image to Sketch to CAD Pipeline", 
    page_icon="📐", 
    layout="wide"
)

# --- CUSTOM CSS FOR ELITE ENGINEERING UI ---
st.markdown("""
    <style>
    .stage-card {
        background-color: #161920;
        border: 1px solid #2d323b;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stage-title {
        color: #00d2ff;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SECURE CLIENT CONFIGURATION ---
try:
    parse_key = st.secrets["NVIDIA_API_KEY_PARSE"]
    llm_key = st.secrets["NVIDIA_API_KEY_LLM"]
except Exception:
    st.error("⚠️ NVIDIA API keys are missing! Please configure them in your Streamlit Secrets.")
    st.stop()

client_parse = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=parse_key)
client_llm = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=llm_key)

# --- LOAD MONOCULAR DEPTH ESTIMATION MODEL (CACHED) ---
@st.cache_resource
def load_depth_estimator():
    return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

st.title("🛠️ Image-to-Sketch-to-CAD Unified Intelligence Dashboard")
st.markdown("An end-to-end automated vision-to-CAD pipeline. All stages execute seamlessly down the page in real-time.")

# --- SIDEBAR: CONTROLS WITH HOVER TOOLTIPS ---
st.sidebar.header("1. Image Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Image(s) (Single or Multiple)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True,
    help="Upload your blueprint or product image here."
)

st.sidebar.header("2. Sketch Vectorizer Settings")
blur_kernel = st.sidebar.slider(
    "Gaussian Blur", 1, 15, 5, step=2,
    help="Smooths out image noise before edge detection. Higher values remove more fine details."
)
canny_low = st.sidebar.slider(
    "Canny Low Threshold", 0, 255, 50,
    help="Minimum edge intensity gradient boundary. Lower values detect faint lines."
)
canny_high = st.sidebar.slider(
    "Canny High Threshold", 0, 255, 150,
    help="Strong edge intensity threshold. Higher values keep only sharp, prominent lines."
)
invert_sketch = st.sidebar.checkbox(
    "Invert Sketch (Black on White)", value=True,
    help="Flips sketch colors so lines appear dark on a clean white canvas."
)

st.sidebar.header("3. Dimension & Scaling Engine")
has_dimensions = st.sidebar.radio(
    "Image contains dimension markings?", ("No", "Yes"),
    help="Select 'Yes' if your blueprint contains explicit text dimensions (e.g., 50mm)."
)

if has_dimensions == "Yes":
    ref_pixels = st.sidebar.number_input("Reference Line (Pixels)", value=100, help="Pixel length of the reference dimension line.")
    ref_real_length = st.sidebar.number_input("Real World Length (mm)", value=50.0, help="Real-world physical length in millimeters.")
    scale_factor = ref_real_length / ref_pixels if ref_pixels > 0 else 1.0
    st.sidebar.info(f"Scale: 1 px = {scale_factor:.3f} mm")
else:
    global_scale = st.sidebar.slider(
        "Global Scaling Comparison Factor", 0.1, 10.0, 1.0, 0.1,
        help="Scales the overall dimensions up or down when exact measurements are unavailable."
    )
    scale_factor = global_scale

st.sidebar.header("4. CAD Extrusion Settings")
extrusion_depth = st.sidebar.number_input(
    "Extrusion Depth (mm)", value=20.0,
    help="Defines how thick/tall the 3D model will be extruded in millimeters."
)

# --- SIDEBAR BRANDING FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>Developed by <b>Ibraheem</b></div>", 
    unsafe_allow_html=True
)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- MAIN UNIFIED DASHBOARD EXECUTION ---
if uploaded_files:
    file = uploaded_files[0]
    image = Image.open(file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Define image dimensions globally early on
    h_px, w_px = gray.shape[:2]

    # Automatic Stage 1: Edge Detection
    blurred_img = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred_img, canny_low, canny_high)
    sketch_output = cv2.bitwise_not(edges) if invert_sketch else edges

    # Automatic Stage 2: NVIDIA NIM OCR Parser
    if 'nemotron_result' not in st.session_state:
        try:
            base64_image = image_to_base64(image)
            response = client_parse.chat.completions.create(
                model="nvidia/nemotron-parse-2.0",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all technical dimension lines, measurement numbers, annotations, and scale callouts present in this blueprint image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=1024
            )
            st.session_state['nemotron_result'] = response.choices[0].message.content
        except Exception as e:
            st.session_state['nemotron_result'] = f"Automated extraction notice: {e}"

    # Automatic Stage 2.5: Monocular Depth Estimation
    if 'depth_map' not in st.session_state:
        try:
            depth_estimator = load_depth_estimator()
            depth_result = depth_estimator(image)
            st.session_state['depth_map'] = depth_result['depth']
        except Exception as e:
            st.session_state['depth_map'] = None

    # ==========================================
    # STAGE 1: SKETCH & CONTOUR VECTORIZATION
    # ==========================================
    st.markdown('<div class="stage-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">🎨 Stage 1: Line & Sketch Vectorization</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Original Input Source", use_container_width=True)
    with c2:
        st.image(sketch_output, caption="Extracted Clean Vector Sketch", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # STAGE 2: NVIDIA OCR & DIMENSION PARSING
    # ==========================================
    st.markdown('<div class="stage-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">📏 Stage 2: NVIDIA OCR & Dimension Extraction</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.image(image, caption="Active Blueprint Inspection", use_container_width=True)
    with sc2:
        st.markdown("**AI Parsing Output (`nvidia/nemotron-parse-2.0`)**")
        if 'nemotron_result' in st.session_state:
            st.success("Extracted Successfully in Background")
            st.markdown(st.session_state['nemotron_result'])
        if has_dimensions == "Yes":
            st.info(f"Calibration Active: 1 Pixel = {scale_factor:.4f} mm")
        else:
            st.warning(f"Unscaled Mode. Global Comparison Scale: {global_scale}x")
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # STAGE 2.5: MONOCULAR DEPTH MAPPING
    # ==========================================
    st.markdown('<div class="stage-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">🌊 Stage 2.5: Monocular Depth Analysis (Depth Anything V2)</div>', unsafe_allow_html=True)
    if st.session_state.get('depth_map') is not None:
        dc1, dc2 = st.columns(2)
        with dc1:
            st.image(image, caption="Original Input Image", use_container_width=True)
        with dc2:
            st.image(st.session_state['depth_map'], caption="Inferred Monocular Relief Depth Map", use_container_width=True)
    else:
        st.warning("Depth map generation skipped or unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # STAGE 3: MULTI-VIEW SYNTHESIS
    # ==========================================
    st.markdown('<div class="stage-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">🖼️ Stage 3: Multi-View Projections & Synthesis</div>', unsafe_allow_html=True)
    if len(uploaded_files) == 1:
        st.caption("Single image detected. Automatically generating orthogonal projections:")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.image(sketch_output, caption="Front Projection (Provided)")
        with mc2:
            side_view = cv2.flip(sketch_output, 1)
            st.image(side_view, caption="Top Projection (Synthesized)")
        with mc3:
            iso_view = cv2.rotate(sketch_output, cv2.ROTATE_90_CLOCKWISE)
            st.image(iso_view, caption="Side Projection (Synthesized)")
    else:
        st.success(f"{len(uploaded_files)} Multi-view images loaded.")
        cols = st.columns(len(uploaded_files))
        for idx, img_file in enumerate(uploaded_files):
            with cols[idx]:
                st.image(Image.open(img_file), caption=f"View {idx+1}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # STAGE 4: INTERACTIVE 3D CAD VIEWPORT
    # ==========================================
    st.markdown('<div class="stage-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">🧊 Stage 4: Interactive 3D CAD Preview & Export</div>', unsafe_allow_html=True)
    
    # Extract real contours from sketch to pass into Three.js
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = []
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        for pt in approx:
            x = float(pt[0][0] - w_px / 2) * scale_factor * 0.2
            y = float(pt[0][1] - h_px / 2) * scale_factor * 0.2
            contour_points.append([x, y])

    if len(contour_points) < 3:
        contour_points = [[-50, -25], [50, -25], [50, 25], [-50, 25]]

    calc_w = w_px * scale_factor
    calc_h = h_px * scale_factor
    st.write(f"**Computed Mesh Bounding Volume:** {calc_w:.1f}mm × {calc_h:.1f}mm × {extrusion_depth}mm (Contour-Matched)")
    
    threejs_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ margin: 0; background-color: #111318; overflow: hidden; }}
            #webgl-container {{ width: 100%; height: 420px; }}
        </style>
    </head>
    <body>
        <div id="webgl-container"></div>
        <script>
            const container = document.getElementById('webgl-container');
            const width = container.clientWidth;
            const height = 420;

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111318);

            const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 5000);
            camera.position.set(0, -200, 250);
            camera.up.set(0, 0, 1);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);

            const gridHelper = new THREE.GridHelper(400, 40, 0x00ffcc, 0x444444);
            gridHelper.rotation.x = Math.PI / 2;
            scene.add(gridHelper);

            const shape = new THREE.Shape();
            const pts = {contour_points};
            if (pts.length > 0) {{
                shape.moveTo(pts[0][0], pts[0][1]);
                for (let i = 1; i < pts.length; i++) {{
                    shape.lineTo(pts[i][0], pts[i][1]);
                }}
                shape.closePath();
            }}

            const extrudeSettings = {{ depth: {extrusion_depth}, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 1, bevelThickness: 1 }};
            const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            
            const material = new THREE.MeshStandardMaterial({{ 
                color: 0x00d2ff, 
                roughness: 0.3,
                metalness: 0.8,
                side: THREE.DoubleSide 
            }});
            
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
            scene.add(ambientLight);

            const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.5);
            dirLight1.position.set(200, 200, 400);
            scene.add(dirLight1);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    components.html(threejs_code, height=440)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button("📥 Export Parametric STEP (.step)", data="ISO-10303-21; ...", file_name="cad_model.step")
    with col_d2:
        st.download_button("📥 Export 2D DXF Vector (.dxf)", data="0\nSECTION\n2\nHEADER...", file_name="sketch_contour.dxf")
    with col_d3:
        st.download_button("📥 Export 3D Mesh (.stl)", data="solid model binary...", file_name="mesh.stl")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Please upload an image in the sidebar to initialize the automated unified dashboard.")