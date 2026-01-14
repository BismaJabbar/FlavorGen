# flavorgenlangchain.py

import pandas as pd
import numpy as np
import re
import ast
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableLambda, RunnableSequence

from difflib import get_close_matches
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
recipes_df = pd.read_csv("recipes_data.csv")

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
You are an expert flavor chemist and chef specializing in molecular gastronomy.

USER FLAVOR INTENT:
"{query}"

FLAVOR GRAPH:
{graph}

RELATED RECIPES:
{recipes}

Generate ONE coherent recipe grounded in aroma chemistry.
Do NOT hallucinate ingredients.
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

