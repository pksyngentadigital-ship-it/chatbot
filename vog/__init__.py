"""Voice of Grower — grower-feedback analytics over a Pinecone index.

Layered so each piece can be tested without the ones above it:

    catalog   data only: products, crops, categories, stopwords
    parsing   text -> tags (crops, products, months, categories)
    planner   question -> QueryPlan
    retrieval QueryPlan -> Evidence
    compose   Evidence -> prompt, or a deterministic reply
    exports   Evidence -> CSV / XLSX / PPTX bytes
    llm       Groq wrapper

Nothing here imports pandas or numpy: together they were ~95MB, which is
most of a serverless bundle budget, for two DataFrame round-trips that
openpyxl and the csv module do directly.
"""
