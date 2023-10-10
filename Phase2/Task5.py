import numpy as np
import streamlit as st

from MongoDB.MongoDBUtils import *


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2"
mod_path = Path(__file__).parent.parent
caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')


k = st.number_input('Enter k for similar images',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

feature_model = st.selectbox(
        "Select Feature Space",
        ("Color Moments", "Histograms of Oriented Gradients(HOG)", "ResNet-AvgPool-1024","ResNet-Layer3-1024","ResNet-FC-1000"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

dimred = st.selectbox(
        "Select Dimensionality Reduction Technique",
        ("SVD", "NNMF", "LDA","k-Means"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

if st.button("Run", type="primary"):

    with st.spinner('Calculating...'):
        
        ### Creating Label-Label Sim Matx -1
        sim_matrix = get_labels_similarity_matrix(feature_model, odd_feature_collection, feature_collection, similarity_collection, caltech101) ##Labels should be in increasing order
        ### Dim reduction on Sim matx -2
        latent_semantics, top_k_indices = get_reduced_dim_labels(sim_matrix, dimred, k) 
        ### Storing latent Semantics - 3
        np.savez(str(mod_path)+"/LatentSemantics/latent_semantics_{feature_model}_{dimred}_{k}.npz", latent_semantics = latent_semantics)
        ### Listing Label Weight Pairs - 4
        list_label_weight_pairs(top_k_indices, latent_semantics)

    st.success('Done!')

