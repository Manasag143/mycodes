In process query :- User query : Kindly provide Region wise Idustry wise count of customers with HIGH Risk Score for 15 Jan 2025
    Identifying Dimensions group name and level name......................
 
/data/fulkrum/text2sql/cube_query_v3.py:240: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-chroma package and should be used instead. To use it run `pip install -U :class:`~langchain-chroma` and import as `from :class:`~langchain_chroma import Chroma``.
  load_embedding_dim = Chroma(persist_directory=persist_directory_dim, embedding_function=self.embedding)
/data/fulkrum/text2sql/cube_query_v3.py:274: LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  result = qa_chain({"query": query, "context": ensemble_retriever})
