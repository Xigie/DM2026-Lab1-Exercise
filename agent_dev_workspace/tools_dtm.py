from typing import List, Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1]
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from the text column in session.dataframe.

        Args:
            max_features: Maximum number of terms to retain in the vocabulary sorted by term frequency.
            ngram_range: List of two integers [min_n, max_n] for the lower and upper boundary of n-grams.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe with 'text' column found in session."}

        ngram_tuple = (ngram_range[0], ngram_range[1])

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple
        )
        texts = session.dataframe["text"].fillna("").tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state & required artifact for downstream tools
        session.artifacts["count_vectorizer"] = vectorizer
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={"max_features": max_features, "ngram_range": ngram_range},
            summary=summary
        )

        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the document-term matrix.

        Args:
            top_n: Number of top most frequent terms to include in the returned summary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Call build_dtm_tool first."}

        # Calculate sum per column across all documents
        column_sums = session.feature_matrix.sum(axis=0).A1.tolist()

        # Build term frequency mapping sorted by count descending
        sorted_pairs = sorted(
            zip(session.feature_names, [int(c) for c in column_sums]),
            key=lambda x: x[1],
            reverse=True
        )
        all_frequencies = {term: count for term, count in sorted_pairs}

        # Cap terms in summary if vocabulary exceeds 50 terms
        top_terms = dict(list(all_frequencies.items())[:top_n])
        summary_frequencies = all_frequencies if len(all_frequencies) <= 50 else top_terms

        # Full DataFrame report for visualization/storage
        full_df = pd.DataFrame(
            list(all_frequencies.items()),
            columns=["term", "frequency"]
        )

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(all_frequencies),
            "top_n": top_n,
            "term_frequencies": summary_frequencies
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_df
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap of a slice of the raw Document-Term Matrix (DTM).

        Args:
            n_terms: Number of terms (columns) to slice from the DTM.
            n_documents: Number of documents (rows) to slice from the DTM.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Call build_dtm_tool first."}

        # Determine slice bounds
        max_docs, max_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, max_docs)
        actual_terms = min(n_terms, max_terms)

        # Slice sub-matrix and convert to dense list
        sliced_matrix = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        sliced_terms = session.feature_names[:actual_terms]
        doc_labels = [f"Doc_{i}" for i in range(actual_docs)]

        # Generate plot and attach to session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"DTM Slice ({actual_docs} docs x {actual_terms} terms)")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_terms,
            "matrix_slice": sliced_matrix.tolist()
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
