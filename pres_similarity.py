import numpy as np
import streamlit as st
import pandas as pd
import hmac
import cryptpandas as crp

st.set_page_config(
    page_title="Presentation Similarity",
    page_icon=":material/category_search:",
    layout="wide",
)

def calculate_cluster_similarities(similarity_matrix, labels):
    """
    Calculate average similarity of each document with others in its cluster
    
    Returns:
    - document_similarities: Average similarity of each document with its cluster
    - cluster_avg_similarities: Average similarity for each cluster
    """
    n_samples = len(labels)
    document_similarities = np.zeros(n_samples)
    cluster_avg_similarities = {}
    
    for i in range(n_samples):
        # Get indices of other documents in the same cluster
        cluster_idx = np.where(labels == labels[i])[0]
        cluster_idx = cluster_idx[cluster_idx != i]  # Exclude self
        
        if len(cluster_idx) > 0:  # If there are other documents in the cluster
            # Calculate average similarity with other documents in cluster
            document_similarities[i] = np.mean(similarity_matrix[i, cluster_idx])
        
    # Calculate average similarity for each cluster
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_docs = np.where(cluster_mask)[0]
        
        if len(cluster_docs) > 1:
            cluster_similarities = []
            for doc in cluster_docs:
                other_docs = cluster_docs[cluster_docs != doc]
                avg_sim = np.mean(similarity_matrix[doc, other_docs])
                cluster_similarities.append(avg_sim)
            cluster_avg_similarities[cluster] = np.mean(cluster_similarities)
        else:
            cluster_avg_similarities[cluster] = 0.0
            
    return document_similarities, cluster_avg_similarities

st.title("Presentation Similarity Exploration Tool")

# Password Check to unlock abstracts.
def password_entered():
    """Checks whether a password entered by the user is correct."""
    if hmac.compare_digest(st.session_state["password"], st.secrets["access_password"]):
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store the password.
    else:
        st.session_state["password_correct"] = False

# Show input for password.
st.text_input(
    "Password to view abstracts", type="password", on_change=password_entered, key="password"
)
if "password_correct" in st.session_state:
    # Returns True if the password is validated.
    if st.session_state.get("password_correct", False):
        st.success("Abstracts Unlocked")
    else:
        st.error("😕 Password incorrect")

st.write("**NOTE:** The titles and abstracts reflect the versions that organizers had available during session organization. The Technical Community and Sessions indicate where session organizers placed each presentation. ***The titles and abstracts are not the final versions as presented at the conference. They do not represent the final presented research at ASABE AIM 2024.***")

# Load DataFrames
# Returns True if the password is validated.
if st.session_state.get("password_correct", False):
    #the_salt = b' \x1e\x9e\xb8%xt\xe5*\x03\x03\xb8<`\x06\xa3'
    df_presentations = crp.read_encrypted(path='encrypted_df.crypt', password=st.secrets['df_password'])
    #df_presentations = pd.read_pickle("df_pres_full.pkl")
else:
    df_presentations = pd.read_pickle("df_pres_basic.pkl")

model = st.radio("Select embedding model to use:",
                 ['nomic-embed-text-v1.5',
                  'cde-small-v1',
                  'all-mpnet-base-v2',
                  'all-MiniLM-L6-v2'],
                  horizontal=True,
                  index = 0)
if model == 'all-MiniLM-L6-v2':
    df_similarity = pd.read_pickle("miniLM_similarities_oral.pkl")
elif model == 'all-mpnet-base-v2':
    df_similarity = pd.read_pickle("mpnet_similarities_oral.pkl")
elif model == 'cde-small-v1':
    df_similarity = pd.read_pickle("cde_similarities_oral.pkl")
else:
    df_similarity = pd.read_pickle("nomic_similarities_oral.pkl")
st.write("nomic-embed-text-v1.5 is the default model. Others are provided for comparison.")
with st.expander("Model Information"):
    st.write("Each model has differences in embedding to capture meaning and in the amount of text that they can process. The nomic-embed-text-v1.5 model processes the entire title and abstract. The cde-small-v1 model has the highest quality scores.")
    st.markdown("""
        |                                                                             | nomic-embed-text-v1.5 | cde-small-v1 | all-mpnet-base-v2 | all-MiniLM-L6-v2 |
        |-----------------------------------------------------------------------------|-----------------------|--------------|-------------------|------------------|
        | **Maximum Input Tokens**                                                    | 8192                  | 512          | 384               | 256              |
        | **[MTEB Clustering Score](https://huggingface.co/spaces/mteb/leaderboard)** |     43.93             |     48.32    |     43.69         |     41.94        |""")
    st.write("For oral presentations at ASABE AIM 2024, the Title and Abstract length distribution (in tokens):") 
    st.markdown("""
        | Minimum | 25th Percentile | Median | Mean | 75th Percentile | Maximum |
        |---------|-----------------|--------|------|-----------------|---------|
        | 84      | 307             | 379.5  | 404  | 473             | 1010    |""")

with st.expander("How similarity scores are calculated"):
    st.write("**Presentation-Session Similarity:** This *presentation metric* is the average cosine similarity between a presentation and all others in its assigned session. It measures how similar a presentation is to others in its session.")
    st.write("**Session Similarity:** This *session metric* is the average cosine similarity between all presentations assigned to the same session. It is an overall indicator of how well a session focuses on one topic.")
    st.write("**Session Std Dev:** This *session metric* is the standard deviation of the Presentation-Session Similarity scores of the presentations assigned to that session. Measures the variation in Presentation-Session scores.  Session Similarity is a better measure of focus, but this metric can be used to identify sessions with outlier presentations.")
    st.write("**Raw Deviation:** This *presentation metric* is the difference between a presentation's Presentation-Session Similarity and its session's Session Similarity. A direct measure of the difference in similarity of a presentation and its session.")
    st.write("**Standardized Deviation:** This *presentation metric* is the average cosine similarity between all presentations assigned to the same session. This standardizes the similarity difference based on the variabilty in a session. It is most useful for identifying single presentations that stand out from an otherwise very focused session.")
pres_session_similarity, session_similarity = calculate_cluster_similarities(df_similarity.to_numpy(), np.array(df_presentations['Original Session']))
df_presentations['Presentation-Session Similarity'] = pres_session_similarity
df_presentations['Session Similarity'] = df_presentations['Original Session'].map(session_similarity)

# Calculate standard deviation for each 'Original Session'
df_presentations['Session Std Dev'] = df_presentations.groupby('Original Session')['Presentation-Session Similarity'].transform('std')
# Calculate deviation from the mean
df_presentations['Raw Deviation'] = df_presentations['Presentation-Session Similarity'] - df_presentations['Session Similarity']
# Calculate standardized deviation
df_presentations['Standardized Deviation'] = df_presentations['Raw Deviation'] / df_presentations['Session Std Dev']

# Create the Sessions DataFrame
df_sessions = df_presentations[['Original Session', 'Session Similarity', 'Session Std Dev']].copy()
# Drop duplicate rows, keeping only the first instance, then reindex.
df_sessions.drop_duplicates(subset=['Original Session'], keep='first', inplace=True)
df_sessions = df_sessions.reset_index(drop=True)

tab_pres, tab_session, tab_edit =  st.tabs(['View Presentations','View Sessions','Edit Placement'])
with tab_pres:
    st.header("Presentations") 
    with st.expander('**Instructions** Click to expand'):
        st.write("Select a presentation by clicking on the checkbox. You can sort the presentation list or search as well.")
        st.write("Once a presentation is selected, its abstract and the ten most similar presentations will appear in a list below.")
        st.write("If you move your mouse over the table, a menu will appear in the top left corner that lets you search within the table or download. Clicking on columns will let you sort by the column too.")
        st.write("If text is cut off, click twice on an cell to see the full text. You can scroll left-right and up-down in the table.")
        st.write("Similarity scores range from 0.0 (not similar) to 1.0 (identical).")
        st.write("The leftmost column is a checkbox column. Click to select a presentation. This may blend with the background on dark themes.")

    event = st.dataframe(
            df_presentations,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Session Similarity" : None,
                'Session Std Dev': None,
                "Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                "Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )


    if event.selection.rows: # Check if a presentation has been selected.
        st.header("Selected Presentation:")
        selected_pres = df_presentations.iloc[event.selection.rows]  # Create a dataframe from the selected presentation row.
        st.write(selected_pres.iloc[0]['Title'])  # It is necessary to request the first row, [0], since it is a dataframe and not just one entry.
        st.header("Most Similar Presentations")
        similar_presentations = df_similarity.loc[selected_pres.iloc[0].name].sort_values(ascending=False) # Create a Series with the  most similar presentations
        # Remove the selected presentation itself from the similar presentations
        similar_presentations = similar_presentations.drop(selected_pres.iloc[0].name)
        # Build the similarity dataframe. Add the similarity score and similarity rank to the dataframe and show it.
        similar_df = df_presentations.loc[similar_presentations.index]
        similar_df.insert(0, "Similarity Score", similar_presentations)
        similar_df.insert(0, "Similarity Rank", np.arange(1,similar_df.shape[0]+1))
        st.dataframe(
            similar_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity" : None,
                "Session Similarity" : None,
                'Session Std Dev': None,
                "Raw Deviation" : None,
                "Standardized Deviation" : None,
            },
            )
with tab_session:
    st.header("Sessions")
    with st.expander('**Instructions** Click to expand'):
        st.write("Select a session by clicking on the checkbox in the leftmost column. Its details and assigned presentations will appear below. You can sort the session list by any column or search for a session name. Just click on the column or mouse over the table.")
    event_session = st.dataframe(
            df_sessions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Session Std Dev" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )

    if event_session.selection.rows: # Check if a session has been selected.
        st.header('Session Details')
        selected_session_df = df_sessions.iloc[event_session.selection.rows]  # Create a dataframe from the selected session row.
        selected_session =selected_session_df.iloc[0]['Original Session']
        st.subheader(selected_session)
        st.write(f"**Session Similarity:** {selected_session_df.iloc[0]['Session Similarity']:.3f}")
        df_selected_session = df_presentations[df_presentations['Original Session'] == selected_session]
        if 'Abstract' in df_selected_session: # Check if the dataframe has the abstract in it to determine how to display.
            st.dataframe(
                df_selected_session,
                use_container_width=True,
                hide_index=True,
                column_order=["Presentation-Session Similarity", 'Standardized Deviation', 'Abstract ID', 'Title', 'Abstract', ],
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                    "Session Similarity" : None,
                    'Session Std Dev': None,
                    "Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    "Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                },
            )
        else:
            st.dataframe(
                df_selected_session,
                use_container_width=True,
                hide_index=True,
                column_order=["Presentation-Session Similarity", 'Standardized Deviation', 'Abstract ID', 'Title', ],
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                    "Session Similarity" : None,
                    'Session Std Dev': None,
                    "Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    "Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                },
            )
with tab_edit:
    with st.expander('**Instructions** Click to expand'):
        st.write("You can reassign presentations to sessions in this table. Just double click on the session name in the \"Assigned Session\" Column. You can change the name to any text. This tool groups all the presentations that share the same session name after each edit. Then it calculates the similarities and creates the session lists and tables below.")
        st.write("**CHANGES ARE NOT SAVED!** Every time the browser is refreshed all the changes are reset. To record your changes, click the download button on the table menu that appears in the upper left corner when your mouse is over the table.")
        st.write("**CHANGES ARE NOT SAVED!** Changing the embedding model using the radio buttons also resets all changes as the new similaries are loaded.")
        st.write("Hint: You can copy and paste entire rows from a spreadsheet program. You can download the list and edit the sessions in a spreadsheet and then paste it back into the webpage to see how it impacts the results. You can also just use a spreadsheet to save your progress and then paste it back into the column to reload where you were. When pasting, make sure the tables are sorted the same way! The Abstract ID can be useful for this.")
    st.header("Editable Presentation List")
    if 'Abstract' in df_presentations: # Check if the dataframe has the abstract in it to determine how to display.
        df_edited = st.data_editor(
            df_presentations,
            use_container_width=True,
            hide_index=True,
            column_order=['Original Session',
                        'Abstract ID',
                        'First Name',
                        'Last Name',
                        'Title',
                        'Abstract',
                        'Original Technical Community',
            ],
            column_config={
                'Original Session' : 'Assigned Session',
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity" : None,
                "Session Similarity" : None,
                'Session Std Dev': None,
                "Raw Deviation" : None,
                "Standardized Deviation" : None,
            },
            disabled=['Abstract ID',
                    'First Name',
                    'Last Name',
                    'Title',
                    'Abstract',
                    'Original Technical Community',
                    'Presentation-Session Similarity',
                    'Session Similarity',
                    'Session Std Dev',
                    'Raw Deviation',
                    'Standardized Deviation',
            ]
        )
    else:
        df_edited = st.data_editor(
            df_presentations,
            use_container_width=True,
            hide_index=True,
            column_order=['Original Session',
                        'Abstract ID',
                        'Title',
                        'Original Technical Community',
            ],
            column_config={
                'Original Session' : 'Assigned Session',
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity" : None,
                "Session Similarity" : None,
                'Session Std Dev': None,
                "Raw Deviation" : None,
                "Standardized Deviation" : None,
            },
            disabled=['Abstract ID',
                    'Title',
                    'Original Technical Community',
                    'Presentation-Session Similarity',
                    'Session Similarity',
                    'Session Std Dev',
                    'Raw Deviation',
                    'Standardized Deviation',
            ]
        )
    edit_pres_session_similarity, edit_session_similarity = calculate_cluster_similarities(df_similarity.to_numpy(), np.array(df_edited['Original Session']))
    df_edited['Presentation-Session Similarity'] = edit_pres_session_similarity
    df_edited['Session Similarity'] = df_edited['Original Session'].map(edit_session_similarity)

    # Calculate standard deviation for each 'Original Session'
    df_edited['Session Std Dev'] = df_edited.groupby('Original Session')['Presentation-Session Similarity'].transform('std')
    # Calculate deviation from the mean
    df_edited['Raw Deviation'] = df_edited['Presentation-Session Similarity'] - df_edited['Session Similarity']
    # Calculate standardized deviation
    df_edited['Standardized Deviation'] = df_edited['Raw Deviation'] / df_edited['Session Std Dev']
    # Create the Edited Sessions DataFrame
    df_sessions_edited = df_edited[['Original Session', 'Session Similarity', 'Session Std Dev']].copy()
    # Drop duplicate rows, keeping only the first instance, then reindex.
    df_sessions_edited.drop_duplicates(subset=['Original Session'], keep='first', inplace=True)
    df_sessions_edited = df_sessions_edited.reset_index(drop=True)

    st.header("Presentations")
    with st.expander("See impact of changes made above on presentations"):
        st.dataframe(
                df_edited,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                    "Session Similarity" : None,
                    'Session Std Dev': None,
                    "Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    "Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                },
        )
    st.header("Sessions")
    with st.expander("Select a session in this table."):
        st.write("Select a session and details for it will appear below this table.")
        st.write("Everytime the presentations are edited, the sessions values are recalculated. It may be necessary to reselect the session after edits.")
        event_edit_session = st.dataframe(
            df_sessions_edited,
            use_container_width=True,
            hide_index=False,
            column_config={
                "Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Session Std Dev" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
            key='edit_session_select'
        )
        if event_edit_session.selection.rows: # Check if a session has been selected.
            st.header('Session Details')
            selected_session_edit_df = df_sessions_edited.iloc[event_edit_session.selection.rows]  # Create a dataframe from the selected session row.
            selected_session_edit =selected_session_edit_df.iloc[0]['Original Session']
            st.subheader(selected_session_edit)
            st.write(f"**Session Similarity:** {selected_session_edit_df.iloc[0]['Session Similarity']:.3f}")
            df_selected_session_edit = df_edited[df_edited['Original Session'] == selected_session_edit]
            if 'Abstract' in df_selected_session_edit: # Check if the dataframe has the abstract in it to determine how to display.
                st.dataframe(
                    df_selected_session_edit,
                    use_container_width=True,
                    hide_index=True,
                    column_order=["Presentation-Session Similarity", 'Standardized Deviation', 'Abstract ID', 'Title', 'Abstract', ],
                    column_config={
                        'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                        "Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                        "Session Similarity" : None,
                        'Session Std Dev': None,
                        "Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                        "Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    },
                    key='edit_session_view'
                )
            else:
                st.dataframe(
                    df_selected_session_edit,
                    use_container_width=True,
                    hide_index=True,
                    column_order=["Presentation-Session Similarity", 'Standardized Deviation', 'Abstract ID', 'Title', ],
                    column_config={
                        'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                        "Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                        "Session Similarity" : None,
                        'Session Std Dev': None,
                        "Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                        "Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    },
                    key='edit_session_view'
                )
