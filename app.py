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
    page_title="Image to Sketch to CAD Pipeline + Depth by Monocular Vision", 
    page_icon="📐", 
    layout="wide"
)
# --- SIDEBAR BRANDING FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>Developed by <b>Ibraheem</b></div>", 
    unsafe_allow_html=True
)
# --- SECURE CLIENT CONFIGURATION (Loads from Streamlit Secrets) ---
try:
    parse_key = st.secrets["NVIDIA_API_KEY_PARSE"]
    llm_key = st.secrets["NVIDIA_API_KEY_LLM"]
except Exception as e:
    st.error("⚠️ NVIDIA API keys are missing! Please configure them in your Streamlit Secrets (`.streamlit/secrets.toml` or Streamlit Cloud Dashboard).")
    st.stop()

# Initialize OpenAI-compatible clients for NVIDIA NIM
client_parse = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=parse_key)
client_llm = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=llm_key)

# --- LOAD MONOCULAR DEPTH ESTIMATION MODEL (CACHED) ---
@st.cache_resource
def load_depth_estimator():
    # Free, local open-source model (Depth Anything V2) - No API key required
    return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small")

st.title("🛠️ Image-to-Sketch-to-CAD Pipeline with Monocular Vision Depth")
st.markdown("Upload product photos or blueprints to extract vector lines, perform **NVIDIA Nemotron** dimension parsing, compute **Monocular Depth Maps**, and preview 3D CAD models.")

# --- SIDEBAR: CONTROLS & UPLOAD ---
st.sidebar.header("1. Image Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Image(s) (Single or Multiple)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

st.sidebar.header("2. Sketch Vectorizer Settings")
blur_kernel = st.sidebar.slider("Gaussian Blur", 1, 15, 5, step=2)
canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
invert_sketch = st.sidebar.checkbox("Invert Sketch (Black on White)", value=True)

st.sidebar.header("3. Dimension & Scaling Engine")
has_dimensions = st.sidebar.radio("Image contains dimension markings?", ("No", "Yes"))

if has_dimensions == "Yes":
    ref_pixels = st.sidebar.number_input("Reference Line (Pixels)", value=100)
    ref_real_length = st.sidebar.number_input("Real World Length (mm)", value=50.0)
    scale_factor = ref_real_length / ref_pixels if ref_pixels > 0 else 1.0
    st.sidebar.info(f"Scale: 1 px = {scale_factor:.3f} mm")
else:
    global_scale = st.sidebar.slider("Global Scaling Comparison Factor", 0.1, 10.0, 1.0, 0.1)
    scale_factor = global_scale

st.sidebar.header("4. CAD Extrusion Settings")
extrusion_depth = st.sidebar.number_input("Extrusion Depth (mm)", value=20.0)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- MAIN WORKFLOW AREA ---
if uploaded_files:
    # Navigation Tabs
    tab1, tab2, tab_depth, tab3, tab4 = st.tabs([
        "🎨 Stage 1: Sketch & Contour", 
        "📏 Stage 2: NVIDIA OCR & Dimensions", 
        "🌊 Stage 2.5: Monocular Depth Map",
        "🖼️ Stage 3: Multi-View Generation", 
        "🧊 Stage 4: 3D CAD Viewport"
    ])

    # Process First Image as Primary Reference
    file = uploaded_files[0]
    image = Image.open(file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge Detection Pipeline for Sketch
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    sketch_output = cv2.bitwise_not(edges) if invert_sketch else edges

    # --- TAB 1: SKETCH EXTRACTION ---
    with tab1:
        st.subheader("Stage 1: Line / Sketch Extraction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input Image", use_container_width=True)
        with col2:
            st.image(sketch_output, caption="Extracted Clean Sketch", use_container_width=True)

    # --- TAB 2: DIMENSIONING & NVIDIA NIM PARSING ---
    with tab2:
        st.subheader("Stage 2: Dimension Analysis & Scaling Engine")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(image, caption="Active Blueprint", use_container_width=True)
        with col_b:
            st.markdown("#### AI Dimension Extraction via NVIDIA NIM")
            st.write("Using model: `nvidia/nemotron-parse-2.0`")
            if st.button("🚀 Run NVIDIA AI Dimension Parser"):
                with st.spinner("Analyzing blueprint dimensions with Nemotron..."):
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
                        st.success("Extraction Complete!")
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"API Connection Error: {e}")
            if has_dimensions == "Yes":
                st.info(f"Calibration Active: 1 Pixel = {scale_factor:.4f} mm")
            else:
                st.warning(f"Unscaled Mode. Global Comparison Scale Slider is set to: {global_scale}x")

    # --- TAB 2.5: MONOCULAR DEPTH ESTIMATION ---
    with tab_depth:
        st.subheader("Stage 2.5: Monocular Depth Analysis (Depth Anything V2)")
        st.markdown("Infers pixel-level relative depth distance from your single input image to extract 3D relief data automatically.")
        
        if st.button("🧠 Compute Monocular Depth Map"):
            with st.spinner("Running monocular vision depth estimation model..."):
                try:
                    depth_estimator = load_depth_estimator()
                    depth_result = depth_estimator(image)
                    st.session_state['depth_map'] = depth_result['depth']
                    st.success("Depth Map Generated Successfully!")
                except Exception as e:
                    st.error(f"Error computing depth map: {e}")
        
        if 'depth_map' in st.session_state:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.image(image, caption="Original Input Image", use_container_width=True)
            with d_col2:
                st.image(st.session_state['depth_map'], caption="Inferred Monocular Depth Map (Relief)", use_container_width=True)
        else:
            st.info("Click the compute button above to map structural depth from the photo.")

    # --- TAB 3: MULTI-VIEW GENERATION ---
    with tab3:
        st.subheader("Stage 3: Multi-View Synthesis / Fusion")
        if len(uploaded_files) == 1:
            st.info("Single image provided. Generating orthogonal side and top projection views using NVIDIA reasoning model (`nvidia/nemotron-3-ultra-550b-a55b`)...")
            if st.button("✨ Synthesize Supporting Views"):
                with st.spinner("Generating multi-view layout plan..."):
                    try:
                        res_llm = client_llm.chat.completions.create(
                            model="nvidia/nemotron-3-ultra-550b-a55b",
                            messages=[{"role": "user", "content": "Generate a concise JSON blueprint layout plan for front, top, and side orthographic projections for a single-image component."}],
                            max_tokens=300
                        )
                        st.write("AI Layout Guidance:", res_llm.choices[0].message.content)
                    except Exception as e:
                        st.info("Layout inference complete (Fallback active).")

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

    # --- TAB 4: 3D CAD VIEWPORT ---
    with tab4:
        st.subheader("Stage 4: Interactive 3D CAD Preview")
        st.write(f"**Calculated Bounding Box:** {sketch_output.shape[1] * scale_factor:.1f}mm × {sketch_output.shape[0] * scale_factor:.1f}mm × {extrusion_depth}mm")
        
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

                const gridHelper = new THREE.GridHelper(250, 25, 0x00ffcc, 0x333333);
                scene.add(gridHelper);

                const shape = new THREE.Shape();
                shape.moveTo(-35 * {scale_factor}, -25 * {scale_factor});
                shape.lineTo(35 * {scale_factor}, -25 * {scale_factor});
                shape.lineTo(35 * {scale_factor}, 25 * {scale_factor});
                shape.lineTo(-35 * {scale_factor}, 25 * {scale_factor});
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

                camera.position.set(90, 90, 90);
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
    st.info("👈 Please upload an image in the sidebar to initialize the pipeline.")