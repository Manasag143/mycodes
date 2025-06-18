# OLD (around line 240)
from langchain_community.vectorstores.chroma import Chroma

# NEW - Add this import at the top
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores.chroma import Chroma


# OLD
result = qa_chain({"query": query, "context": ensemble_retriever})

# NEW
result = qa_chain.invoke({"query": query, "context": ensemble_retriever})


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Around line 240 - UPDATE THIS:
load_embedding_dim = Chroma(persist_directory=persist_directory_dim, embedding_function=self.embedding)

# Around line 274 - UPDATE THIS:
result = qa_chain.invoke({"query": query, "context": ensemble_retriever})  # Changed from __call__ to invoke



# Find similar lines in get_measures method and update:
result = qa_chain.invoke({"query": query, "context": ensemble_retriever})  # Instead of qa_chain()
