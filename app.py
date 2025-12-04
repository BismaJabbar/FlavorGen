import streamlit as st
import pandas as pd


st.set_page_config(page_title="Taste & Recipe Generator", layout="wide")


st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #FFFFFF;  /* White background */
    }

    /* Main title */
    h1 { color: #4B0082 !important; }  /* Indigo / dark purple */

    /* Markdown subheadings */
    .purple-subheader {
        color: #800080;  /* Purple */
        font-size: 24px;
        font-weight: bold;
    }

    /* Body text */
    .stMarkdown, .stMarkdown p {
        color: #000000 !important;  /* Black text */
    }

    /* Text input box */
    .stTextInput>div>div>input {
        background-color: #FFFFFF !important;  
        color: #000000 !important;            
        border: 2px solid #800080 !important; 
    }

    /* Button */
    .stButton>button {
        background-color: #800080 !important; 
        color: #FFFFFF !important;            
        font-weight: bold;
    }

    /* Dataframe table text */
    .stDataFrame table {
        color: #000000 !important;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🌸 Taste & Recipe Generator")
st.write("Enter your desired flavor and get molecules, ingredients, and recipes!")

# User Input
user_query = st.text_input("Enter a flavor or dish (e.g., citrus cake):")

# Button action
if st.button("Generate") and user_query.strip() != "":

    st.info("🔎 Searching flavor molecules...")

    
    molecules_data = {
        "Compound Name": ["Limonene", "Citral", "Vanillin"],
        "Flavors": ["citrus, fresh", "citrus, lemon", "sweet, creamy"]
    }
    molecules_df = pd.DataFrame(molecules_data)

    
    st.markdown('<p class="purple-subheader">🧪 Matched Flavor Molecules</p>', unsafe_allow_html=True)
    st.dataframe(molecules_df)

    
    ingredients_data = {
        "Ingredient Name": ["Lemon zest", "Orange extract", "Vanilla extract"],
        "Category": ["Citrus", "Citrus", "Sweet"],
        "Molecules Matched": [["limonene"], ["citral"], ["vanillin"]]
    }
    ingredients_df = pd.DataFrame(ingredients_data)

    st.markdown('<p class="purple-subheader">🥗 Suggested Ingredients</p>', unsafe_allow_html=True)
    st.dataframe(ingredients_df)

   
    st.markdown('<p class="purple-subheader">🍰 Generated Recipe</p>', unsafe_allow_html=True)
    st.markdown("""
**Citrus Cake Recipe**  
- Ingredients: Lemon zest, Orange extract, Vanilla extract  
- Steps:  
  1. Mix dry ingredients.  
  2. Add wet ingredients and citrus extracts.  
  3. Bake at 180°C for 30 mins.  
- Why it works: Citrus molecules provide fresh flavor, vanilla adds sweetness and depth.
""")
