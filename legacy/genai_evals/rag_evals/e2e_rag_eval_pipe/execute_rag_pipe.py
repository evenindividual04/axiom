from rag_evals.e2e_rag_eval_pipe.image2markdown import img2markdown_main
from rag_evals.e2e_rag_eval_pipe.ingest2vectorstore import ingest_data_to_store_with_fetch
from rag_evals.e2e_rag_eval_pipe.pdf2image import pdf2image_main

if __name__ == "__main__":
    pdf2image_main()
    img2markdown_main()
    ingest_data_to_store_with_fetch()