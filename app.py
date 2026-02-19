import os
import textwrap
import streamlit as st
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

# ---- Disease Info Database ----
disease_info = {
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "cause": [
            "Bacterial spot is caused by several species of gram-negative bacteria in the genus Xanthomonas.",
            "In culture, these bacteria produce yellow, mucoid colonies. A 'mass' of bacteria can be observed oozing from a lesion by making a cross-sectional cut through a leaf lesion, placing the tissue in a droplet of water, placing a cover-slip over the sample, and examining it with a microscope (~200X)."
        ],
        "treatment": [
            "The primary management strategy of bacterial spot begins with use of certified pathogen-free seed and disease-free transplants.",
            "The bacteria do not survive well once host material has decayed, so crop rotation is recommended. Once the bacteria are introduced into a field or greenhouse, the disease is very difficult to control.",
            "Plants are routinely sprayed with copper-containing bactericides to maintain a 'protective' cover on the foliage and fruit."
        ]
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "cause": [
            "Early blight is caused by the fungus Alternaria solani.",
            "The fungus overwinters in infected plant debris and soil, and can be transmitted by wind, water, and insects."
        ],
        "treatment": [
            "Use resistant tomato varieties if available.",
            "Rotate crops and avoid planting tomatoes in the same soil for at least two years.",
            "Apply fungicides such as chlorothalonil or copper-based sprays."
        ]
    },
    "Tomato___Late_blight": {
         "crop": "Tomato",
         "disease": "Late Blight",
         "cause": [
             "Late blight is caused by the oomycete pathogen Phytophthora infestans.",
             "It thrives in cool, wet weather and can spread rapidly, destroying entire crops if left unchecked."
         ],
         "treatment": [
             "Plant resistant varieties.",
             "Destroy infected plants immediately to prevent spread.",
             "Apply fungicides like mancozeb or chlorothalonil, especially during cool, wet weather."
         ]
    },
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "cause": [
            "Caused by the fungus Venturia inaequalis.",
            "The fungus overwinters in fallen leaves and releases spores in the spring during wet weather."
        ],
        "treatment": [
            "Plant resistant apple varieties.",
            "Rake and destroy fallen leaves to reduce the source of infection.",
            "Apply fungicides such as captan or myclobutanil at the first sign of disease."
        ]
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot",
        "cause": [
            "Caused by the fungus Botryosphaeria obtusa.",
            "The pathogen infects fruit, leaves, and bark, often entering through wounds."
        ],
        "treatment": [
            "Prune out dead or diseased wood.",
            "Remove and destroy mummified fruit from the tree and ground.",
            "Apply fungicides during the growing season."
        ]
    },
     "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "cause": [
            "Caused by the fungus Gymnosporangium juniperi-virginianae.",
            "Requires two hosts: apple trees and Eastern red cedar (or junipers) to complete its life cycle."
        ],
        "treatment": [
            "Remove nearby cedar or juniper trees if possible.",
            "Plant resistant apple varieties.",
            "Apply fungicides like immunox or sulfur during the spring."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn",
        "disease": "Common Rust",
        "cause": [
            "Caused by the fungus Puccinia sorghi.",
            "Spores are windblown and can travel long distances to infect corn plants."
        ],
        "treatment": [
            "Plant resistant corn hybrids.",
            "Fungicides may be necessary in severe cases or for sweet corn."
        ]
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "cause": [
            "Caused by the fungus Alternaria solani.",
            "Similar to tomato early blight, it survives in plant debris and soil."
        ],
        "treatment": [
            "Plant resistant potato varieties.",
            "Practice crop rotation.",
            "Apply fungicides preventatively."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "cause": [
             "Caused by Phytophthora infestans.",
             "This is the same pathogen that caused the Irish Potato Famine."
        ],
        "treatment": [
             "Plant certified disease-free seed potatoes.",
             "Eliminate cull piles and volunteer potatoes.",
             "Apply fungicides regularly during the growing season."
        ]
    }
}

def get_disease_info(class_name):
    if class_name in disease_info:
        return disease_info[class_name]
    
    # Generic fallback
    if "___" in class_name:
        parts = class_name.split("___")
        crop = parts[0]
        disease = parts[1].replace("_", " ")
    else:
        parts = class_name.split("_")
        crop = parts[0]
        disease = " ".join(parts[1:])
        
    return {
        "crop": crop,
        "disease": disease.title(),
        "cause": [
            f"The specific cause for this {disease} on {crop} is currently being researched.",
            "It may be caused by fungal, bacterial, or viral pathogens common to this crop."
        ],
        "treatment": [
            "Isolate the affected plant to prevent spread.",
            "Remove infected leaves or parts.",
            "Consult a local agricultural extension expert for specific chemical or organic treatments."
        ]
    }

# ---- Page Configuration ----
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS ----
st.markdown("""
    <style>
    /* Global Styles */
    html {
        scroll-behavior: smooth;
    }
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #f0fdf4 !important; /* Very light green fallback */
        background-image: url('https://images.unsplash.com/photo-1542601906990-b4d3fb7d5b43?q=80&w=2670&auto=format&fit=crop') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Navbar Styles */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: white;
        padding: 1rem 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 99999;
    }
    .navbar-brand {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1b4332;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .navbar-links {
        display: flex;
        gap: 2rem;
    }
    .navbar-links a {
        color: #4b5563;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s;
    }
    .navbar-links a:hover, .navbar-links a.active {
        color: #2ecc71;
    }
    .navbar-links a.active {
        border-bottom: 2px solid #2ecc71;
        padding-bottom: 2px;
    }
    .btn-get-started {
        background-color: #2ecc71;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        text-decoration: none;
        font-weight: bold;
        transition: background-color 0.3s;
        border: none;
        cursor: pointer;
    }
    .btn-get-started:hover {
        background-color: #27ae60;
    }
    
    /* Adjust main content padding because of fixed navbar */
    .block-container {
        padding-top: 6rem !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #1b4332; /* Dark green for text */
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Cards */
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    /* Custom Classes */
    .highlight {
        color: #2ecc71;
        font-weight: bold;
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;} /* Hide default streamlit header */
    </style>
    """, unsafe_allow_html=True)
    
# ---- Pages ----


def landing_page():
    # Navbar
    st.markdown("""
    <div class="navbar">
        <div class="navbar-brand">
            🌿 Plant Disease Detection App
        </div>
        <div class="navbar-links">
            <a href="#home" class="active" target="_self">Home</a>
            <a href="#about" target="_self">About</a>
            <a href="#services" target="_self">Services</a>
            <a href="#contact" target="_self">Contact</a>
        </div>
        <div class="navbar-buttons">
            <a href="?modal=true" class="btn-get-started" target="_self">🚀 Get Started</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Section (Replacing the active tool on Landing Page)
    st.markdown("""
    <div id="home" style="margin-top: 5rem; padding: 5rem 2rem; text-align: center; scroll-margin-top: 100px;">
        <h1 style="font-size: 3.5rem; font-weight: 800; color: #1b4332; margin-bottom: 2rem;">
            Protect Your Crops with AI
        </h1>
        <p style="font-size: 1.25rem; color: #555; max-width: 800px; margin: 0 auto 3rem auto; line-height: 1.6;">
            Early detection is key to healthy harvests. Our advanced AI-powered tool identifies plant diseases in seconds, helping you take action before it's too late.
        </p>
        <div style="display: flex; gap: 1.5rem; justify-content: center;">
            <a href="?modal=true" target="_self" style="background-color: #2ecc71; color: white; padding: 1rem 2rem; border-radius: 50px; text-decoration: none; font-weight: bold; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3); transition: transform 0.2s;">
                Start Detection 🚀
            </a>
            <a href="#about" target="_self" style="background-color: white; color: #1b4332; padding: 1rem 2rem; border-radius: 50px; text-decoration: none; font-weight: bold; font-size: 1.1rem; border: 2px solid #e8f5e9; transition: background 0.2s;">
                Learn More
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- About Section ----
    st.markdown("""
<div id="about" style="margin-top: 5rem; padding: 3rem; background-color: #f9f9f9; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); scroll-margin-top: 100px;">
    <h2 style="font-size: 2.5rem; font-weight: 800; color: #1b4332; margin-bottom: 1.5rem; text-align: center;">
        About Plant Disease Detection
    </h2>
    <div style="max-width: 900px; margin: 0 auto; color: #555; font-size: 1.1rem; line-height: 1.8;">
        <p style="margin-bottom: 1.5rem;">
            This application represents a leap forward in precision agriculture, harnessing the power of advanced <strong>Artificial Intelligence</strong> to protect global food security. By bridging the gap between traditional farming wisdom and cutting-edge technology, we provide farmers and agriculturists with an instant, pocket-sized expert. The system not only identifies diseases but also provides actionable, sustainable treatment recommendations to ensure healthier crops and higher yields.
        </p>
        <p>
            At its core, the app detects plant diseases using state-of-the-art <strong>Deep Learning</strong> models, specifically <strong>Vision Transformers (ViT)</strong> and <strong>Swin Transformers</strong>. Unlike traditional Convolutional Neural Networks (CNNs), these attention-based architectures capture global context and intricate patterns on leaf surfaces, allowing for superior accuracy even in complex field conditions. Trained on the extensive <strong>PlantVillage dataset</strong> covering 38 distinct disease classes across 14 crop species, our models achieve exceptional precision, ensuring that no symptom goes unnoticed.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

    # ---- Services Section ----
    st.markdown("""
<div id="services" style="margin-top: 5rem; padding-bottom: 5rem; text-align: center; scroll-margin-top: 100px;">
<h2 style="font-size: 3rem; font-weight: 800; color: #1b4332; margin-bottom: 3rem;">Our Services</h2>
<div style="display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap;">
<!-- Service Card 1 -->
<div style="background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); padding: 3rem 2rem; width: 350px; text-align: left; transition: transform 0.3s ease; border-top: 5px solid #2ecc71;">
<div style="width: 64px; height: 64px; margin-bottom: 1.5rem;">
<svg viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 100%; height: 100%;">
<path d="M2 22s0-13 11-13 11 13 11 13M9 22V9M12 9V5c0-1.7-1.3-3-3-3s-3 1.3-3 3" />
<path d="M7 16c2.2-2 5.5-2 8 0" />
</svg>
</div>
<h3 style="font-size: 1.5rem; font-weight: 700; color: #1b4332; margin-bottom: 1rem;">Plant Disease Detection</h3>
<p style="color: #666; font-size: 1rem; line-height: 1.6; margin-bottom: 2rem;">
Real-time plant disease identification using computer vision and machine learning. Upload images of plant leaves to get instant diagnosis and treatment recommendations for various plant diseases.
</p>
<div style="display: inline-flex; align-items: center; background-color: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 50px; font-weight: 600; font-size: 0.875rem;">
<span style="font-size: 1.2rem; margin-right: 6px;">●</span>
97.93% Accuracy
</div>
</div>
<!-- Service Card 2 -->
<div style="background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); padding: 3rem 2rem; width: 350px; text-align: left; transition: transform 0.3s ease; border-top: 5px solid #3b82f6;">
<div style="width: 64px; height: 64px; margin-bottom: 1.5rem;">
<svg viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 100%; height: 100%;">
<line x1="18" y1="20" x2="18" y2="10"></line>
<line x1="12" y1="20" x2="12" y2="4"></line>
<line x1="6" y1="20" x2="6" y2="14"></line>
</svg>
</div>
<h3 style="font-size: 1.5rem; font-weight: 700; color: #1b4332; margin-bottom: 1rem;">Crop Health Monitoring</h3>
<p style="color: #666; font-size: 1rem; line-height: 1.6; margin-bottom: 2rem;">
Track the long-term health of your crops with historical data analysis. Monitor trends and get predictions to prevent outbreaks before they happen.
</p>
<div style="display: inline-flex; align-items: center; background-color: #dbeafe; color: #1e40af; padding: 6px 12px; border-radius: 50px; font-weight: 600; font-size: 0.875rem;">
<span style="font-size: 1.2rem; margin-right: 6px;">●</span>
24/7 Monitoring
</div>
</div>
<!-- Service Card 3 -->
<div style="background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); padding: 3rem 2rem; width: 350px; text-align: left; transition: transform 0.3s ease; border-top: 5px solid #f59e0b;">
<div style="width: 64px; height: 64px; margin-bottom: 1.5rem;">
<svg viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 100%; height: 100%;">
<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
<circle cx="9" cy="7" r="4"></circle>
<path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
<path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
</svg>
</div>
<h3 style="font-size: 1.5rem; font-weight: 700; color: #1b4332; margin-bottom: 1rem;">Expert Consultation</h3>
<p style="color: #666; font-size: 1rem; line-height: 1.6; margin-bottom: 2rem;">
Get direct access to agricultural experts for difficult cases. Upload your queries and get personalized advice from certified agronomists.
</p>
<div style="display: inline-flex; align-items: center; background-color: #fef3c7; color: #92400e; padding: 6px 12px; border-radius: 50px; font-weight: 600; font-size: 0.875rem;">
<span style="font-size: 1.2rem; margin-right: 6px;">●</span>
Certified Experts
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ---- Contact Section ----
    st.markdown("""
<div id="contact" style="margin-top: 5rem; padding-bottom: 5rem; scroll-margin-top: 100px;">
<h2 style="font-size: 3rem; font-weight: 800; color: #1b4332; margin-bottom: 3rem; text-align: center;">Get in Touch</h2>
<div style="display: flex; gap: 3rem; justify-content: center; flex-wrap: wrap;">
<!-- Contact Info Card -->
<div style="background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); padding: 3rem; width: 400px; height: fit-content;">
<h3 style="font-size: 1.5rem; font-weight: 700; color: #1b4332; margin-bottom: 2rem;">Contact Information</h3>
<div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
<div style="width: 40px; height: 40px; margin-right: 15px; display: flex; align-items: center; justify-content: center; background-color: #e8f5e9; border-radius: 50%;">
<svg viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 20px; height: 20px;">
<path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
<polyline points="22,6 12,13 2,6"></polyline>
</svg>
</div>
<span style="color: #555; font-size: 1rem;">info@plantdisease.com</span>
</div>
<div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
<div style="width: 40px; height: 40px; margin-right: 15px; display: flex; align-items: center; justify-content: center; background-color: #e8f5e9; border-radius: 50%;">
<svg viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 20px; height: 20px;">
<circle cx="12" cy="12" r="10"></circle>
<polyline points="12 6 12 12 16 14"></polyline>
</svg>
</div>
<span style="color: #555; font-size: 1rem;">Mon - Fri: 9:00 AM - 6:00 PM</span>
</div>
<div style="display: flex; align-items: center;">
<div style="width: 40px; height: 40px; margin-right: 15px; display: flex; align-items: center; justify-content: center; background-color: #e8f5e9; border-radius: 50%;">
<svg viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 20px; height: 20px;">
<circle cx="12" cy="12" r="10"></circle>
<line x1="2" y1="12" x2="22" y2="12"></line>
<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
</svg>
</div>
<span style="color: #555; font-size: 1rem;">www.plantdisease.com</span>
</div>
</div>
<!-- Contact Form Card -->
<div style="background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); padding: 3rem; width: 400px;">
<form action="">
<div style="margin-bottom: 1.5rem;">
<label style="display: block; color: #1b4332; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">Full Name</label>
<input type="text" style="width: 100%; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; outline: none; transition: border 0.3s; font-size: 1rem; color: #374151;">
</div>
<div style="margin-bottom: 1.5rem;">
<label style="display: block; color: #1b4332; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">Email Address</label>
<input type="email" style="width: 100%; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; outline: none; transition: border 0.3s; font-size: 1rem; color: #374151;">
</div>
<div style="margin-bottom: 1.5rem;">
<label style="display: block; color: #1b4332; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">Subject</label>
<input type="text" style="width: 100%; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; outline: none; transition: border 0.3s; font-size: 1rem; color: #374151;">
</div>
<div style="margin-bottom: 2rem;">
<label style="display: block; color: #1b4332; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">Message</label>
<textarea style="width: 100%; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; height: 120px; resize: none; outline: none; transition: border 0.3s; font-size: 1rem; color: #374151; font-family: inherit;" placeholder="Tell us how we can help you..."></textarea>
</div>
<button type="button" style="width: 100%; background-color: #2ecc71; color: white; padding: 14px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: background 0.3s; font-size: 1rem;">
Send Message
</button>
</form>
</div>
</div>
</div>
""", unsafe_allow_html=True)


def detection_page():
    # Back to Home Button
    if st.button("← Back to Home"):
        st.session_state.service = None
        st.rerun()

    st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem; color: #1b4332; font-size: 3rem; font-weight: 800;'>Plant Disease Detection (v2.0)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 3rem; color: #666; font-size: 1.1rem;'>Upload an image of your plant's leaves and get instant AI-powered disease diagnosis with treatment recommendations.</p>", unsafe_allow_html=True)

    # Main Content Layout
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Styled Upload Card
        st.markdown("""
        <div style="background: white; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); padding: 2rem; border: 1px solid #e5e7eb; margin-bottom: 20px;">
            <h3 style="text-align: center; color: #1b4332; margin-bottom: 1.5rem;">Upload Plant Image</h3>
            <div style="border: 2px dashed #2ecc71; border-radius: 15px; padding: 2rem; text-align: center; background-color: #f0fdf4;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">☁️</div>
                <p style="color: #2ecc71; font-weight: bold; font-size: 1.2rem;">Drop your image here</p>
                <p style="color: #666; font-size: 0.9rem;">or click to browse from your device</p>
            </div>
            <div style="display: flex; gap: 10px; justify-content: center; margin-top: 1rem;">
                 <span style="background:#e8f5e9; padding: 4px 8px; border-radius: 4px; font-size: 12px; color: #2ecc71; font-weight: bold;">JPG</span>
                 <span style="background:#e8f5e9; padding: 4px 8px; border-radius: 4px; font-size: 12px; color: #2ecc71; font-weight: bold;">PNG</span>
                 <span style="background:#e8f5e9; padding: 4px 8px; border-radius: 4px; font-size: 12px; color: #2ecc71; font-weight: bold;">WEBP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"], label_visibility="collapsed")
        
        # Model Selection (Hidden or styled better?) - Keeping functional
        st.markdown("### Model Configuration")
        model_options = {"vit": "Vision Transformer (ViT)", "swin": "Swin Transformer"}
        model_choice = st.selectbox("Select Model Architecture", list(model_options.keys()), format_func=lambda x: model_options[x])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Check if selected model exists
            model_path = f"checkpoints/best_model_{model_choice}.pth"
            if not os.path.exists(model_path):
                st.warning(f"⚠️ The {model_options[model_choice]} model has not been trained yet. Please run `python train.py --model {model_choice}` in your terminal.")
                st.button("🔍 Analyze Plant Disease", disabled=True, use_container_width=True)
            else:
                if st.button("🔍 Analyze Plant Disease", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing..."):
                        model = load_model(model_choice)
                    if model:
                        transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5]*3,[0.5]*3)
                        ])
                        img_tensor = transform(image).unsqueeze(0)
                        
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            _, pred = torch.max(outputs, 1)
                        
                        prediction_name = CLASS_NAMES[pred.item()]
                        st.balloons()
                        st.success(f"**Result=** {prediction_name}")
                        
                        # Get detailed info
                        info = get_disease_info(prediction_name)
                        
                        # Display Analysis Result Card (Yellow)
                        st.markdown(f"""
<div class="card" style="background-color: #fff9c4; border: 1px solid #fbc02d; color: #333; margin-top: 20px;">
    <h4 style="color: #f57f17; display: flex; align-items: center; gap: 10px; margin-top: 0;">
        📋 Analysis Result:
    </h4>
    <p><strong>Crop:</strong> {info['crop']}</p>
    <p><strong>Disease:</strong> {info['disease']}</p>
    
    <p style="margin-top: 15px; font-weight: bold;">Cause of disease:</p>
    <p style="line-height: 1.6;">
        {' '.join(info['cause'])}
    </p>
</div>
""", unsafe_allow_html=True)
                        
                        # Display Treatment Recommendations
                        st.markdown(f"""
<div style="margin-top: 20px;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <span style="font-size: 2.5rem; color: #2ecc71;">💊</span>
        <h2 style="margin: 0; color: #1b4332;">Treatment Recommendations</h2>
    </div>
    
    <div class="card" style="border-left: 5px solid #2ecc71;">
        <h4 style="color: #2ecc71; margin-top: 0;">✅ Recommended Actions</h4>
        <ul style="padding-left: 20px; line-height: 1.6;">
            {''.join([f'<li>{treatment}</li>' for treatment in info['treatment']])}
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)


    with col_right:
        # Info Column
        st.markdown(textwrap.dedent("""
            <div style="background: white; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); padding: 24px;">
                <div style="margin-bottom: 16px;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="#10b981" xmlns="http://www.w3.org/2000/svg">
                         <path d="M12 2C7.03 2 3 6.03 3 11C3 12.87 3.57 14.61 4.54 16.03C4.2 16.59 4 17.27 4 18C4 20.21 5.79 22 8 22C9.56 22 10.91 21.11 11.58 19.81C11.72 19.82 11.86 19.82 12 19.82C12.14 19.82 12.28 19.82 12.42 19.81C13.09 21.11 14.44 22 16 22C18.21 22 20 20.21 20 18C20 17.27 19.8 16.59 19.46 16.03C20.43 14.61 21 12.87 21 11C21 6.03 16.97 2 12 2ZM8 20C6.9 20 6 19.1 6 18C6 16.9 6.9 16 8 16C9.1 16 10 16.9 10 18C10 19.1 9.1 20 8 20ZM12 17.82C10.74 17.82 9.61 17.24 8.87 16.34C8.6 15.65 8 15.05 7.27 14.68C6.48 14.28 5.76 13.78 5.15 13.18C5.05 12.48 5 11.75 5 11C5 7.13 8.13 4 12 4C15.87 4 19 7.13 19 11C19 11.75 18.95 12.48 18.85 13.18C18.24 13.78 17.52 14.28 16.73 14.68C16 15.05 15.4 15.65 15.13 16.34C14.39 17.24 13.26 17.82 12 17.82ZM16 20C14.9 20 14 19.1 14 18C14 16.9 14.9 16 16 16C17.1 16 18 16.9 18 18C18 19.1 17.1 20 16 20Z"></path>
                    </svg>
                </div>
                <h3 style="color: #064e3b; font-size: 1.5rem; font-weight: 700; margin: 0 0 12px 0;">AI-Powered Detection</h3>
                <p style="color: #6b7280; font-size: 1rem; line-height: 1.6; margin-bottom: 24px;">
                    Our advanced computer vision model can identify over 38 different plant diseases with high accuracy.
                </p>
                <div style="margin-bottom: 24px; border-bottom: 1px solid #f3f4f6; padding-bottom: 24px;">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <div style="margin-right: 12px; color: #10b981;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                <circle cx="12" cy="12" r="3"></circle>
                            </svg>
                        </div>
                        <span style="color: #4b5563; font-size: 0.95rem;">Image recognition technology</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <div style="margin-right: 12px; color: #10b981;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"></path>
                                <path d="M3 5v14a2 2 0 0 0 2 2h16v-5"></path>
                                <path d="M18 12a2 2 0 0 0 0 4h4v-4Z"></path>
                            </svg>
                        </div>
                        <span style="color: #4b5563; font-size: 0.95rem;">Trained on 87K+ plant images</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="margin-right: 12px; color: #10b981;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                        </div>
                        <span style="color: #4b5563; font-size: 0.95rem;">Results in under 3 seconds</span>
                    </div>
                </div>
                <p style="color: #6b7280; font-size: 0.95rem; margin-bottom: 16px;">Reliable disease classification</p>
                <div style="display: inline-flex; align-items: center; background-color: #10b981; color: white; padding: 8px 16px; border-radius: 9999px; font-weight: 600; font-size: 0.875rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;">
                        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                        <polyline points="17 6 23 6 23 12"></polyline>
                    </svg>
                    97.93% Accuracy
                </div>
            </div>
        """), unsafe_allow_html=True)

        st.markdown(textwrap.dedent("""
                <div style="background: white; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); padding: 2rem; border: 1px solid #e5e7eb; margin-top: 20px;">
                     <h3 style="color: #2ecc71;">Detectable Diseases</h3>
                     <ul style="padding-left: 20px; color: #444; margin-top: 1rem; line-height: 1.6;">
                        <li><strong>Apple:</strong> Scab, Black rot, Rust</li>
                        <li><strong>Tomato:</strong> Blight, Leaf mold, Mosaic virus</li>
                        <li><strong>Corn:</strong> Rust, Leaf blight</li>
                        <li><strong>Potato:</strong> Early/Late blight</li>
                        <li>And many more...</li>
                    </ul>
                </div>
            """), unsafe_allow_html=True)



# ---- Modal Logic ----
@st.dialog("Choose Your Service")
def show_service_modal():
    st.write("Select the AI-powered service you'd like to explore:")
    
    # Plant Disease Detection Button (Primary)
    if st.button("🌿 Plant Disease Detection", use_container_width=True, type="primary"):
        st.session_state.service = "detection"
        st.query_params.clear()
        st.rerun()

# ---- Main Router ----
def main():
    # Handle Modal Logic first
    if st.query_params.get("modal") == "true":
        show_service_modal()

    # Route based on session state
    if st.session_state.get("service") == "detection":
        detection_page()
    else:
        landing_page()


# ---- Model Logic (Cached) ----
@st.cache_resource
def load_model(model_name):
    # Using the same logic as before but wrapped for the new UI
    try:
        if model_name == "vit":
            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=38)
            model.load_state_dict(torch.load("checkpoints/best_model_vit.pth", map_location="cpu"))
        else:
            model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=38)
            model.load_state_dict(torch.load("checkpoints/best_model_swin.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---- Constants ----
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

if __name__ == "__main__":
    main()
