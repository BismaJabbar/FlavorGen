import streamlit as st
from datetime import datetime
import base64

from flavorgenlangchain import (
    generate_flavor,
    molecule_to_ingredients,
    build_aroma_graph,
    select_seed_ingredients,
    flavor_retriever,
    recipe_retriever
)


@st.cache_resource
def get_flavor_retriever():
    from flavorgenlangchain import flavor_retriever
    return flavor_retriever

@st.cache_resource
def get_recipe_retriever():
    from flavorgenlangchain import recipe_retriever
    return recipe_retriever

# Now call them when needed
flavor_retriever = get_flavor_retriever()
recipe_retriever = get_recipe_retriever()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="FlavorGen AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ------------------ SIDEBAR ------------------
st.sidebar.title("FlavorGen AI")

# Sidebar radio updates the session state
selected = st.sidebar.radio(
    "Navigation",
    ["Home", "Flavor Generator", "History", "About"],
    index=["Home", "Flavor Generator", "History", "About"].index(st.session_state.page)
)
st.session_state.page = selected

dark_mode = st.sidebar.toggle("Dark Mode", value=True)

# ------------------ THEME ------------------
def apply_theme(dark):
    if dark:
        bg = "#0f1f1a"
        text = "white"
        card = "rgba(255,255,255,0.15)"
    else:
        bg = "#f4f5f2"
        text = "#1e1e1e"
        card = "rgba(255,255,255,0.85)"

    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    h1,h2,h3,h4,h5,h6,p,span,label {{
        color: {text};
    }}

    .glass {{
        background: {card};
        backdrop-filter: blur(16px);
        padding: 40px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 12px 32px rgba(0,0,0,0.25);
        max-width: 1000px;
        margin: 60px auto;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
    }}
    [data-testid="stSidebar"] * {{
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(dark_mode)

# ------------------ CARD HELPERS ------------------
def card_start():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ HERO VIDEO ------------------
def hero_video(video_file):
    with open(video_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .hero-container {{
        position: relative;
        width: 100%;
        height: 90vh;
        overflow: hidden;
    }}
    .hero-container video {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }}
    .hero-text-wrapper {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.7);
    }}
    .hero-text-top {{
        font-size: 64px;
        font-weight: 700;
        margin-bottom: 20px;
    }}
    .hero-text-bottom {{
        font-size: 32px;
        font-weight: 500;
    }}
    </style>

    <div class="hero-container">
        <video autoplay muted loop>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
        <div class="hero-text-wrapper">
            <div class="hero-text-top">FlavorGen</div>
            <div class="hero-text-bottom">AI + Culinary Artistry</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------ HOME PAGE ------------------
# ------------------ HOME PAGE ----------------
if st.session_state.page == "Home":

    hero_video("background_loop.mp4")

    # About Card
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="font-size:48px; font-weight:700;">FlavorGen AI</h2>
        <p style="font-size:22px; line-height:1.6;">
            FlavorGen AI is an intelligent system that combines 
            <b>aroma chemistry, ingredient compatibility</b>, 
            and <b>generative AI</b> to inspire creative 
            and scientifically grounded recipes. FlavorGen AI blends culinary science and AI innovation, 
            crafting flavor combinations that excite the senses and inspire the next generation of recipes.
            From classic ingredients to futuristic combinations, FlavorGen AI uses AI-driven flavor science
             to turn your culinary ideas into mouthwatering creations
        </p>
    </div>
    """, unsafe_allow_html=True)
    card_end()

    # Why it Matters
    card_start()
    st.markdown('<div style="text-align:center;"><h2 style="font-size:40px; font-weight:700;">Why It Matters</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="text-align:center;"><h3 style="font-size:28px;">Smart Pairings</h3><p>Flavor harmony based on molecular similarity.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align:center;"><h3 style="font-size:28px;">AI Creativity</h3><p>Generate innovative recipe concepts instantly.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="text-align:center;"><h3 style="font-size:28px;">Interactive History</h3><p>Save and refine your flavor ideas.</p></div>', unsafe_allow_html=True)
    card_end()

    # ---------------- GET STARTED BUTTON ----------------
    card_start()

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <h2 style="font-size:40px; font-weight:700;">Get Started</h2>
            <p style="font-size:22px; line-height:1.6;">
                Move to the Flavor Generator to create your first pairing.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # CENTER THE BUTTON USING COLUMNS
    cols = st.columns([1, 2, 1])  # middle column is bigger
    with cols[1]:
        # CUSTOM CSS FOR BIG BUTTON
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            font-size: 24px;
            padding: 20px 60px;
            border-radius: 15px;
            background-color: #2FA67A;
            color: white;
        }
        div.stButton > button:hover {
            background-color: #36C58B;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("Get Started!"):
            st.session_state.page = "Flavor Generator"

    card_end()


# ------------------ FLAVOR GENERATOR ------------------
elif st.session_state.page == "Flavor Generator":
    card_start()
    st.header("Flavor Generator")

    # Initialize query in session state
    if "query" not in st.session_state:
        st.session_state.query = ""

    # Text input for query
    st.session_state.query = st.text_input(
        "Enter flavor query",
        value=st.session_state.query,
        placeholder="e.g., umami & citrus"
    )

    # Button click logic
    if st.button("Generate Recipe") and st.session_state.query.strip():
        query = st.session_state.query  # local variable for convenience

        with st.spinner("Generating..."):

            # ---------------- DEBUG: check intermediate outputs ----------------
            flavors = flavor_retriever.invoke(query)
            st.write("Flavors retrieved:", flavors)

            compounds = [d.metadata["compound"] for d in flavors]
            st.write("Compounds:", compounds)

            ingredients = molecule_to_ingredients(compounds)
            st.write("Ingredients:", ingredients)

            graph = build_aroma_graph(ingredients)
            st.write("Graph:", graph)

            seed_ingredients = select_seed_ingredients(graph, query)
            st.write("Seed ingredients:", seed_ingredients)
            # --------------------------------------------------------------------

            # Now call the full recipe generator
            result_dict = generate_flavor(query)

        st.subheader("Step 1: Ingredients matched")
        st.write(result_dict["ingredients"])

        st.subheader("Step 2: Aroma graph")
        st.json(result_dict["graph"])

        st.subheader("Step 3: Seed Ingredients")
        st.write(result_dict["seed_ingredients"])

        st.subheader("Step 4: Top Recipes")
        st.write(result_dict["top_recipes"])

        st.subheader("Final Recipe")
        st.markdown(result_dict["final_recipe"])

    card_end()


# ------------------ ABOUT ------------------
# ------------------ ABOUT ----------------
elif st.session_state.page == "About":

    # ---------------- PAGE TITLE ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="font-size:52px; font-weight:800;">About FlavorGen AI</h1>
        <p style="font-size:22px; line-height:1.6; max-width:900px; margin:auto;">
            Explore the science of taste with <b>AI-driven flavor generation</b>.
            Our platform merges culinary creativity, molecular chemistry, and AI
            to craft innovative and balanced flavor pairings.
        </p>
    </div>
    """, unsafe_allow_html=True)
    card_end()

    # ---------------- ABSTRACT / PROJECT OVERVIEW ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="font-size:40px; font-weight:700;">Project Overview</h2>
        <p style="font-size:20px; line-height:1.6; max-width:900px; margin:auto;">
            FlavorGen AI is a Final Year Project designed to assist chefs, researchers,
            and food enthusiasts in <b>creating innovative flavor combinations</b>.
            Leveraging <b>aroma chemistry, ingredient compatibility, and generative AI</b>, 
            the system generates balanced and novel recipes, bridging the gap 
            between science and culinary artistry. Through AI, manual recipe creation
            is transformed into an automated, generative process that saves time
            and inspires creativity. The food and beverage industry is continuing to adopt
             AI to analyze consumer preferences, generate new taste profiles and support recipe development. Through AI we are hereby transforming the whole manual system into automated generative system. This process not only generates new recipes but also saves time which takes in manual recipes making and that’s what this project aims to use Generative AI which can support this domain by generating
             new flavours based on flavour profiles, sensory descriptions and existing flavour profiles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    card_end()

    # ---------------- OUR VISION & MISSION ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="font-size:40px; font-weight:700;">Our Vision & Mission</h2>
        <p style="font-size:20px; max-width:900px; margin:auto;">
            <b>Vision:</b> To empower culinary innovation through AI and scientific flavor analysis.<br>
            <b>Mission:</b> To provide a platform that combines <b>science, AI, and creativity</b>
            to generate harmonious and exciting flavor pairings for every recipe.
        </p>
    </div>
    """, unsafe_allow_html=True)
    card_end()

    # ---------------- KEY FEATURES ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="font-size:40px; font-weight:700;">Key Features</h2>
        <ul style="font-size:20px; max-width:800px; margin:auto; text-align:left;">
            <li>AI-driven flavor pairing suggestions based on molecular similarity.</li>
            <li>Interactive history to save and refine generated recipes.</li>
            <li>Creative recipe generation combining aroma chemistry and ingredient compatibility.</li>
            <li>Easy-to-use interface for culinary experimentation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    card_end()

    # ---------------- TEAM / CONTRIBUTORS ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
     <h2 style="font-size:40px; font-weight:700;">Team & Contributors</h2>
     <div style="display:flex; justify-content:center; gap:50px; flex-wrap:wrap; margin-top:20px;">
        <div style="text-align:center;">
            <h3 style="margin-top:10px;">Areeba Madavia</h3>
        </div>
        <div style="text-align:center;">
            <h3 style="margin-top:10px;">Bisma Jabbar</h3>
        </div>
        <div style="text-align:center;">
            <h3 style="margin-top:10px;">Rija Jilani</h3>
        </div>
     </div>
     <div style="text-align:center;">
       <h3 style="font-size:40px; font-weight:700;">Our Respected Supervisor</h3>
       <h3 style="margin-top:10px;">Dr. Shahid Khan</h3>
     </div>

    </div>
    """, unsafe_allow_html=True)
    card_end()


    # ---------------- TESTIMONIALS ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="font-size:40px; font-weight:700;">Testimonials</h2>
        <div style="max-width:900px; margin:auto; margin-top:20px;">
            <p style="font-size:20px; font-style:italic;">
                "FlavorGen AI helped me discover flavor combinations I never would have thought of — truly innovative!"<br>
                <b>- Test User</b>
            </p>
            <p style="font-size:20px; font-style:italic;">
                "A perfect blend of science and creativity. Ideal for modern chefs."<br>
                <b>- Another User</b>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    card_end()

    # ---------------- FUTURE WORK / ROADMAP ----------------
    card_start()
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="font-size:40px; font-weight:700;">Future Work</h2>
        <p style="font-size:20px; line-height:1.6; max-width:900px; margin:auto;">
            Upcoming features include:<br>
            - Real-time recipe suggestions<br>
            - Multi-ingredient flavor analysis<br>
            - Dietary restriction integration<br>
            - Enhanced interactive AI tools for chefs and culinary researchers
        </p>
    </div>
    """, unsafe_allow_html=True)
    card_end()


