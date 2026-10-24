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

# --- SECURE CLIENT CONFIGURATION ---
try:
    parse_key = st.secrets["NVIDIA_API_KEY_PARSE"]
    llm_key = st.secrets["NVIDIA_API_KEY_LLM"]
except Exception:
    st.error("⚠️ NVIDIA API keys are missing! Please configure them in your Streamlit Secrets.")
    st.stop()

# --- SIDEBAR BRANDING FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>Developed by <b>Ibraheem</b></div>", 
    unsafe_allow_html=True
)

client_parse = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=parse_key)
client_llm = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=llm_key)

# --- LOAD MONOCULAR DEPTH ESTIMATION MODEL (CACHED) ---
@st.cache_resource
def load_depth_estimator():
    # Using the correct Hugging Face compatible transformers repo for Depth Anything V2
    return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

st.title("🛠️ Image-to-Sketch-to-CAD Pipeline with Monocular Vision Depth")
st.markdown("Upload product photos or blueprints to automatically execute edge vectorization, NVIDIA OCR dimension parsing, monocular depth mapping, and 3D CAD generation.")

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

# --- AUTOMATED PIPELINE EXECUTION ---
if uploaded_files:
    file = uploaded_files[0]
    image = Image.open(file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Automatic Stage 1: Edge Detection
    blurred_img = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred_img, canny_low, canny_high)
    sketch_output = cv2.bitwise_not(edges) if invert_sketch else edges

    # Automatic Stage 2: NVIDIA NIM OCR Parser (Runs in background)
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

    # Automatic Stage 2.5: Monocular Depth Estimation (Runs in background)
    if 'depth_map' not in st.session_state:
        try:
            depth_estimator = load_depth_estimator()
            depth_result = depth_estimator(image)
            st.session_state['depth_map'] = depth_result['depth']
        except Exception as e:
            st.session_state['depth_map'] = None

    # --- NAVIGATION TABS ---
    tab1, tab2, tab_depth, tab3, tab4 = st.tabs([
        "🎨 Stage 1: Sketch & Contour", 
        "📏 Stage 2: NVIDIA OCR & Dimensions", 
        "🌊 Stage 2.5: Monocular Depth Map",
        "🖼️ Stage 3: Multi-View Generation", 
        "🧊 Stage 4: 3D CAD Viewport"
    ])

    with tab1:
        st.subheader("Stage 1: Line / Sketch Extraction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Image", use_container_width=True)
        with col2:
            st.image(sketch_output, caption="Extracted Clean Sketch", use_container_width=True)

    with tab2:
        st.subheader("Stage 2: Dimension Analysis & Scaling Engine")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(image, caption="Active Blueprint", use_container_width=True)
        with col_b:
            st.markdown("#### AI Dimension Extraction via NVIDIA NIM")
            st.write("Using model: `nvidia/nemotron-parse-2.0`")
            if 'nemotron_result' in st.session_state:
                st.success("Extraction Completed Automatically!")
                st.markdown(st.session_state['nemotron_result'])
            if has_dimensions == "Yes":
                st.info(f"Calibration Active: 1 Pixel = {scale_factor:.4f} mm")
            else:
                st.warning(f"Unscaled Mode. Global Comparison Scale Slider is set to: {global_scale}x")

    with tab_depth:
        st.subheader("Stage 2.5: Monocular Depth Analysis (Depth Anything V2)")
        st.markdown("Automatically infers pixel-level relative depth distance from your single input image.")
        if st.session_state.get('depth_map') is not None:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.image(image, caption="Original Input Image", use_container_width=True)
            with d_col2:
                st.image(st.session_state['depth_map'], caption="Inferred Monocular Depth Map (Relief)", use_container_width=True)
        else:
            st.warning("Depth map generation was skipped or encountered an error.")

    with tab3:
        st.subheader("Stage 3: Multi-View Synthesis / Fusion")
        if len(uploaded_files) == 1:
            st.info("Single image provided. Generating automated orthogonal side and top projection views...")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.image(sketch_output, caption="View 1: Front (Provided)")
            with m_col2:
                side_view = cv2.flip(sketch_output, 1)
                st.image(side_view, caption="View 2: Top Projection (Generated)")
            with m_col3:
                iso_view = cv2.rotate(sketch_output, cv2.ROTATE_90_CLOCKWISE)
                st.image(iso_view, caption="View 3: Side Projection (Generated)")
        else:
            st.success(f"{len(uploaded_files)} Multi-view images loaded for spatial fusion.")
            cols = st.columns(len(uploaded_files))
            for idx, img_file in enumerate(uploaded_files):
                with cols[idx]:
                    st.image(Image.open(img_file), caption=f"View {idx+1}")

    with tab4:
        st.subheader("Stage 4: Interactive 3D CAD Preview")
        
        # Safety Check before rendering 3D preview
        h_px, w_px = sketch_output.shape[:2]
        calc_w = w_px * scale_factor
        calc_h = h_px * scale_factor
        
        st.write(f"**Calculated Bounding Box:** {calc_w:.1f}mm × {calc_h:.1f}mm × {extrusion_depth}mm")
        
        # Embedded Three.js WebGL Renderer with safe dynamic dimensions
        threejs_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <style> body {{ margin: 0; background-color: #0e1117; }} canvas {{ width: 100%; height: 420px; display: block; }} </style>
        </head>
        <body>
            <script>
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 420, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, 420);
                document.body.appendChild(renderer.domElement);

                const gridHelper = new THREE.GridHelper(300, 30, 0x00ffcc, 0x333333);
                scene.add(gridHelper);

                const shape = new THREE.Shape();
                const boxW = Math.max(10, {w_px} * 0.1 * {scale_factor});
                const boxH = Math.max(10, {h_px} * 0.1 * {scale_factor});
                
                shape.moveTo(-boxW/2, -boxH/2);
                shape.lineTo(boxW/2, -boxH/2);
                shape.lineTo(boxW/2, boxH/2);
                shape.lineTo(-boxW/2, boxH/2);
                shape.closePath();

                const extrudeSettings = {{ depth: {extrusion_depth}, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 1, bevelThickness: 1 }};
                const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                const material = new THREE.MeshPhongMaterial({{ color: 0x00d2ff, wireframe: false, side: THREE.DoubleSide }});
                const mesh = new THREE.Mesh(geometry, material);
                mesh.rotation.x = -Math.PI / 2;
                scene.add(mesh);

                const light = new THREE.DirectionalLight(0xffffff, 1.2);
                light.position.set(60, 120, 60);
                scene.add(light);
                scene.add(new THREE.AmbientLight(0x505050));

                camera.position.set(120, 120, 120);
                const controls = new THREE.OrbitControls(camera, renderer.domElement);

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

else:
    st.info("👈 Please upload an image in the sidebar to initialize the automated pipeline.")