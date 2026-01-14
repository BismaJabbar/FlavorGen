# flavorgenlangchain.py

import pandas as pd
import numpy as np
import re
import ast
import json
import csv        # <-- MUST import this
import sys 
import gdown
import os
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableLambda, RunnableSequence


from difflib import get_close_matches

csv.field_size_limit(sys.maxsize)
# ---------------------------------------
# Load models
# ---------------------------------------
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------
# Load data
# ---------------------------------------
flavor_db = pd.read_csv("FlavorDatabase_normalized.csv")
ing_mol_conc = pd.read_csv("ingredient_molecule_normalized_with_conc_synthetic.csv")
aroma = pd.read_csv("scent_compounds_odor_normalized.csv")

FILE_ID = "1PFrGkI4nvr9S-1GfHdfZte1ocq8sUvxc"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
LOCAL_FILE = "recipes_data.csv"

# Download if not already downloaded
if not os.path.exists(LOCAL_FILE):
    gdown.download(URL, LOCAL_FILE, quiet=False)

# Load CSV safely using python engine
recipes_df = pd.read_csv(
    LOCAL_FILE,
    engine='python',           # safer for messy CSV
    on_bad_lines='warn'        # warn and skip bad rows
)


recipes_df["NER_list"] = recipes_df["NER"].apply(ast.literal_eval)

# ---------------------------------------
# Normalization helpers
# ---------------------------------------
def normalize_text_list(text_list):
    normalized = []
    for text in text_list:
        if not text:
            continue
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        normalized.append(text)
    return normalized


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


recipes_df["normalized_ingredients"] = recipes_df["NER_list"].apply(normalize_text_list)
aroma["compound_norm"] = aroma["Compound"].apply(normalize_name)

# ---------------------------------------
# FAISS stores
# ---------------------------------------
flavor_docs = [
    Document(
        page_content=f"{row['flavor']} flavor compound",
        metadata={"compound": row["compound_name"]},
    )
    for _, row in flavor_db.iterrows()
]

flavor_store = FAISS.from_documents(flavor_docs, embeddings)
flavor_retriever = flavor_store.as_retriever(search_kwargs={"k": 10})

recipe_docs = [
    Document(
        page_content=" ".join(row["normalized_ingredients"]),
        metadata={"title": row["title"]},
    )
    for _, row in recipes_df.head(10000).iterrows()
]

recipe_store = FAISS.from_documents(recipe_docs, embeddings)
recipe_retriever = recipe_store.as_retriever(search_kwargs={"k": 5})

# ---------------------------------------
# Helper logic
# ---------------------------------------
def filter_ingredients_by_molecule_q95(df, molecules, quantile=0.95):
    out = []
    for mol in molecules:
        subset = df[df["Molecule"] == mol]
        if subset.empty:
            continue
        q = subset["conc_level"].quantile(quantile)
        out.append(subset[subset["conc_level"] >= q])
    return pd.concat(out) if out else pd.DataFrame()


def molecule_to_ingredients(compounds, cutoff=0.7, quantile=0.95):
    """
    Convert a list of molecule names to ingredients using fuzzy matching.
    compounds: list of strings (molecules from flavor_retriever)
    cutoff: similarity threshold for fuzzy matching (0-1)
    """
    if not compounds:
        return []

    # Normalize molecules in dataframe
    df = ing_mol_conc.copy()
    df["Molecule_norm"] = df["Molecule"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    ingredients = set()

    for comp in compounds:
        comp_norm = comp.lower()
        comp_norm = re.sub(r"[^a-z0-9]", "", comp_norm)  # remove special chars

        # Fuzzy match against dataframe molecules
        matches = get_close_matches(comp_norm, df["Molecule_norm"].tolist(), n=5, cutoff=cutoff)

        for m in matches:
            matched_rows = df[df["Molecule_norm"] == m]
            # Optional: apply quantile filter
            filtered_rows = filter_ingredients_by_molecule_q95(matched_rows, [m], quantile=quantile)
            if not filtered_rows.empty:
                ingredients.update(filtered_rows["Ingredient"].dropna().tolist())
            else:
                ingredients.update(matched_rows["Ingredient"].dropna().tolist())

    return list(ingredients)



def map_molecule_to_aroma(molecule, aroma_list):
    return [a for a in aroma_list if molecule in a or a in molecule]


def build_aroma_graph(ingredients):
    df = ing_mol_conc[ing_mol_conc["Ingredient"].isin(ingredients)].copy()
    df["molecule_norm"] = df["Molecule"].apply(normalize_name)

    aroma_list = aroma["compound_norm"].unique().tolist()

    df["matched_aromas"] = df["molecule_norm"].apply(
        lambda x: map_molecule_to_aroma(x, aroma_list)
    )

    df = df[df["matched_aromas"].map(len) > 0]

    ingredient_to_odors = (
        df.explode("matched_aromas")
        .merge(
            aroma[["compound_norm", "Odor"]],
            left_on="matched_aromas",
            right_on="compound_norm",
            how="inner",
        )
        .groupby("Ingredient")["Odor"]
        .apply(lambda x: set(o.lower() for o in x))
        .to_dict()
    )

    return ingredient_to_odors


def select_seed_ingredients(ingredient_to_odors, query):
    if not ingredient_to_odors:
        return []

    odor_vocab = list({o for odors in ingredient_to_odors.values() for o in odors})

    if not odor_vocab:
        return []

    odor_emb = semantic_model.encode(
        [o + " food aroma" for o in odor_vocab],
        normalize_embeddings=True
    )

    if odor_emb is None or len(odor_emb) == 0:
        return []

    query_emb = semantic_model.encode(
        [query + " taste flavor"],
        normalize_embeddings=True
    )

    scores = cosine_similarity(query_emb, odor_emb)[0]
    matched_odors = {odor_vocab[i] for i, s in enumerate(scores) if s >= 0.6}

    return [
        ing for ing, odors in ingredient_to_odors.items() if odors & matched_odors
    ]

# ---------------------------------------
# Prompt + LLM
# ---------------------------------------
prompt = PromptTemplate(
    input_variables=["query", "graph", "recipes"],
    template="""
You are an expert flavor chemist and chef specializing in molecular gastronomy,
aroma chemistry, and data-driven recipe design.

Your decisions MUST be grounded in the provided flavor similarity graph.
Flavor compatibility is computed from shared aroma compounds and odor descriptors.

Higher similarity values indicate stronger molecular flavor harmony.

---------------------------------------
STRICT CONSTRAINTS (MANDATORY)
---------------------------------------

1. Use ONLY ingredients present in the provided flavor graph or synergy ingredient list.
2. Prefer ingredient pairings with similarity score >= 0.6.
3. Do NOT invent new flavor pairings or unsupported combinations.
4. Basic culinary staples are allowed (salt, oil, water, sugar, flour, eggs, butter).
5. The recipe MUST emphasize the user's target flavor intent.
6. The dish type must be chosen logically to best express the flavor chemistry.
7. Do NOT force baking, cooking, or techniques that do not fit the dish type.

---------------------------------------
USER FLAVOR INTENT
---------------------------------------

"{query}"

---------------------------------------
FLAVOR GRAPH CONTEXT
---------------------------------------

This graph encodes ingredient compatibility using shared aroma descriptors.
Edges represent flavor similarity strength.

{graph}

---------------------------------------
RELATED RECIPES
---------------------------------------

{recipes}

---------------------------------------
TASK OBJECTIVES
---------------------------------------

1. Analyze the flavor graph and identify the strongest ingredient pairings.
2. Select ONE optimal ingredient cluster for a coherent dish.
3. Decide the most suitable dish category (sauce, soup, drink, glaze, marinade, dessert, cooked dish, etc.).
4. Generate ONE complete, well-balanced recipe.
5. Clearly explain the molecular flavor logic behind the pairing.

---------------------------------------
OUTPUT FORMAT (STRICT)
---------------------------------------

Follow the adaptive template below.
Include ONLY sections that make sense for the chosen dish type.

# {{Recipe Title}}

**Category:** {{Dish type}}
**Servings/Yield:** {{Yield}}
**Total Time:** {{Time}}
{{Optional: **Cooking/Baking Temperature:** {{Temperature}} }}

---

## Ingredients & Flavor / Molecular Rationale

Group ingredients logically (Base, Aromatics, Synergy Ingredients, Seasoning, etc.).

For each ingredient:

- **{{Ingredient}} — {{Amount}}**
  *Explain its molecular contribution, aroma role, or synergy logic*

### Synergy Ingredients (From Graph)
Use ingredients from `{graph}` and justify their inclusion.

- **{{Synergy Ingredient}} — {{Amount}}**
  *Why this pairing strengthens the target flavor*

---

## Instructions (Step-by-Step)

Adapt steps based on dish type:
- Sauces -> simmer, reduce, emulsify
- Drinks -> infuse, blend, shake
- Soups -> sauté, simmer, finish
- Marinades -> whisk, coat, rest
- Cooked dishes -> prep, cook, assemble

1. {{Step 1}}
2. {{Step 2}}
3. {{Step 3}}
...

---

## Flavor Science / Molecular Explanation

Explain:
- Dominant aroma compounds
- Why these ingredients are graph-compatible
- How cooking or preparation affects aroma release
- Balance of acidity, fat, sweetness, umami, or bitterness
- Any Maillard reactions, ester formation, terpene volatility, or emulsification effects

---

## Final Notes
- Texture and aroma profile
- Serving suggestions
- Optional substitutions (if graph-compatible)
- Variations or extensions

IMPORTANT:
This is a data-grounded recipe.
Do NOT hallucinate ingredients, flavors, or chemistry.
Reason step-by-step internally before answering.
"""
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
recipe_chain = prompt | llm

# ---------------------------------------
# Utility
# ---------------------------------------
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    return obj

# ---------------------------------------
# MAIN FUNCTION (Streamlit calls THIS)
# ---------------------------------------

  

def generate_flavor(query: str):
    chain = RunnableSequence(
        # Step 1: Get flavors
        RunnableLambda(lambda q: {"query": q, "flavors": flavor_retriever.invoke(q)})
        
        # Step 2: Extract compounds
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "compounds": [d.metadata["compound"] for d in x["flavors"]]
        })
        
        # Step 3: Ingredients
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "compounds": x["compounds"],
            "ingredients": molecule_to_ingredients(x["compounds"])
        })
        
        # Step 4: Aroma graph
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "ingredients": x["ingredients"],
            "graph": build_aroma_graph(x["ingredients"])
        })
        
        # Step 5: Seed ingredients and top recipes
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "ingredients": x["ingredients"],
            "graph": x["graph"],
            "seed_ingredients": select_seed_ingredients(x["graph"], x["query"]),
            "top_recipes_docs": recipe_retriever.invoke(" ".join(x["graph"].keys()))
        })
        
        # Step 6: Format top recipes and final recipe
        | RunnableLambda(lambda x: {
            "ingredients": x["ingredients"],
            "graph": x["graph"],
            "seed_ingredients": x["seed_ingredients"],
            "top_recipes": "\n".join([d.metadata.get("title","") for d in x["top_recipes_docs"]]),
            "final_recipe": recipe_chain.invoke({
                "query": x["query"],
                "graph": json.dumps(make_json_safe(x["graph"]), indent=2),
                "recipes": "\n".join([d.metadata.get("title","") for d in x["top_recipes_docs"]])
            }).content
        })
    )

    return chain.invoke(query)

