import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import numpy as np

# Import torch and transformers only when needed (not at the top level)
# This helps avoid the PyTorch class path issue with Streamlit's watcher

# Set page title
st.set_page_config(page_title="Acronym Definitions App", layout="wide")

# Custom embedding function using the specified model
class HuggingFaceEmbeddingFunction:
    def __init__(self, model_name="Alibaba-NLP/gte-modernbert-base"):
        # Import here to avoid top-level import issues
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # Store torch module for later use
        self.torch = torch
        
    def __call__(self, texts):
        # Import torch only if needed and not already imported
        if not hasattr(self, 'torch'):
            import torch
            self.torch = torch
            
        # Tokenize the texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Compute token embeddings
        with self.torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Mean pooling to get sentence embeddings
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = self.torch.sum(token_embeddings * input_mask_expanded, 1) / self.torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().numpy()

# Add app description and instructions
st.sidebar.title("About")
st.sidebar.info(
    "This app allows you to upload a CSV file with acronyms and their definitions. "
    "The definitions are stored in a vector database for semantic search capabilities."
)

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload a CSV file with 'acronym' and 'definition' columns
    2. Click 'Process CSV' to load data into the database
    3. Use the search box to find relevant acronyms
    """
)

# Initialize session state variables
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'embedding_function' not in st.session_state:
    st.session_state.embedding_function = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None

def init_db():
    # Add a try-except block to handle potential errors
    try:
        # Initialize embedding function with a progress indicator
        with st.spinner("Loading embedding model (this may take a minute)..."):
            st.session_state.embedding_function = HuggingFaceEmbeddingFunction()
        
        # Initialize Chroma client
        persist_directory = "chroma_db"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        st.session_state.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        try:
            st.session_state.collection = st.session_state.chroma_client.get_collection(
                name="acronyms",
                embedding_function=st.session_state.embedding_function
            )
            st.success("Connected to existing acronym database.")
        except Exception as e:
            st.info("Creating new acronym database.")
            st.session_state.collection = st.session_state.chroma_client.create_collection(
                name="acronyms",
                embedding_function=st.session_state.embedding_function
            )
        
        st.session_state.db_initialized = True
        st.success("Database initialization complete.")
    except Exception as e:
        st.error(f"Error initializing database: {e}")
        # For debugging
        st.exception(e)

def process_csv(df):
    try:
        with st.status("Loading acronyms and definitions into the database...", expanded=True) as status:
            # Initialize the database if not already done
            if not st.session_state.db_initialized:
                st.write("Initializing database...")
                init_db()
            
            if not st.session_state.db_initialized:
                st.error("Database initialization failed. Please check the logs.")
                return
            
            # Process each row and add to the database
            total_rows = len(df)
            batch_size = min(100, total_rows)  # Process in batches to improve performance
            
            # Prepare data for batch processing
            documents = []
            metadatas = []
            ids = []
            
            for i, row in df.iterrows():
                progress = int((i + 1) / total_rows * 100)
                try:
                    acronym = str(row['acronym'])
                    definition = str(row['definition'])
                    
                    # Skip empty entries
                    if not acronym or not definition or pd.isna(acronym) or pd.isna(definition):
                        continue
                    
                    # Add to batch
                    documents.append(definition)
                    metadatas.append({"acronym": acronym})
                    ids.append(f"acronym_{i}")
                    
                    # Update status periodically
                    if (i + 1) % 10 == 0 or i == total_rows - 1:
                        status.update(label=f"Processing: {progress}% complete", state="running")
                    
                    # Process in batches
                    if len(documents) >= batch_size or i == total_rows - 1:
                        if documents:  # Only add if there are documents to add
                            st.session_state.collection.add(
                                documents=documents,
                                metadatas=metadatas,
                                ids=ids
                            )
                            st.write(f"Added batch of {len(documents)} entries")
                            # Clear batch data
                            documents = []
                            metadatas = []
                            ids = []
                except Exception as e:
                    st.warning(f"Error processing row {i}: {e}")
                    continue
            
            status.update(label="All acronyms and definitions loaded successfully!", state="complete")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        st.exception(e)

def search_acronyms(query, n_results=5):
    if not st.session_state.db_initialized:
        st.error("Database not initialized. Please upload a CSV file first.")
        return None
    
    try:
        # Try direct acronym match first
        results = st.session_state.collection.query(
            query_texts=[query],
            where={"acronym": query},
            n_results=n_results
        )
        
        # If no direct matches, do semantic search
        if not results['documents'][0]:
            results = st.session_state.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return None

# Main app layout with tabs
st.title("Acronym Definitions App")

# Create tabs for different functions
tab1, tab2 = st.tabs(["📥 Upload & Process", "🔍 Search"])

with tab1:
    st.header("Upload CSV")
    st.write("Upload a CSV file with 'acronym' and 'definition' columns to create a searchable acronym database.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if the required columns exist
            if 'acronym' in df.columns and 'definition' in df.columns:
                st.success("CSV file loaded successfully!")
                
                # Display a preview of the data
                st.subheader("Preview of the data")
                st.dataframe(df.head())
                
                # Show table info
                st.info(f"CSV contains {len(df)} rows with acronyms and definitions.")
                
                # Process the CSV if not already processed
                col1, col2 = st.columns([1, 2])
                with col1:
                    process_button = st.button("Process CSV", type="primary")
                
                if process_button:
                    process_csv(df)
            else:
                st.error("The CSV file must contain 'acronym' and 'definition' columns.")
                
                # Show actual columns for debugging
                st.write("Found columns:", list(df.columns))
                
                # Suggest potential fixes
                st.info("If your columns have different names or extra spaces, you may need to rename them.")
                
                # Option to view the data anyway
                if st.button("Show data preview anyway"):
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            st.exception(e)

with tab2:
    st.header("Search Acronyms")
    
    # Database status indicator
    if st.session_state.db_initialized:
        st.success("Database is ready for searching")
    else:
        st.warning("Database not initialized. Please upload and process a CSV file first.")
    
    # Search interface
    search_query = st.text_input("Enter an acronym or search term:")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        search_button = st.button("Search", type="primary", disabled=not st.session_state.db_initialized)
        
    with col2:
        n_results = st.slider("Max results", min_value=1, max_value=20, value=5)
    
    if search_query and st.session_state.db_initialized and search_button:
        with st.spinner("Searching..."):
            results = search_acronyms(search_query, n_results)
            
            if results and results['documents'][0]:
                st.subheader("Search Results")
                
                # Display results in a more organized way
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0],
                    results.get('distances', [[0] * len(results['documents'][0])])[0]
                )):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"### {metadata['acronym']}")
                            if 'distances' in results:
                                st.caption(f"Relevance: {100 * (1 - min(distance, 1)):.1f}%")
                        with col2:
                            st.markdown(f"{doc}")
                        st.divider()
            else:
                st.info("No results found.")
                
    # Add a small guide
    with st.expander("Search Tips"):
        st.markdown("""
        - You can search by exact acronym or by description terms
        - The search uses semantic similarity, so related concepts will also be found
        - For best results, use specific and descriptive search terms
        """)
        
# Add a footer
st.markdown("---")
st.caption("Acronym Definitions App powered by ChromaDB and Hugging Face")

# Initialize database button in sidebar for convenience
if not st.session_state.db_initialized:
    if st.sidebar.button("Initialize Database"):
        init_db()
