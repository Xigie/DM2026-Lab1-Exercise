from typing import Dict, Any, List
import numpy as np
import pandas as pd
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags near-constant and low-variance terms for removal across documents.

        Args:
            threshold: Minimum variance threshold. Terms with variance strictly below this value are filtered.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Call build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0, ddof=0)

        # Build term variance DataFrame sorted descending by variance (Column 1: term, Column 2: variance)
        df_variances = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = df_variances[df_variances["variance"] >= threshold]
        removed_df = df_variances[df_variances["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        # Save kept terms to session state
        session.artifacts["filtered_feature_names"] = kept_terms

        total_terms = len(session.feature_names)
        variances_dict = dict(zip(df_variances["term"], [round(float(v), 6) for v in df_variances["variance"]]))

        result_id = session.next_result_id("variance_filter")

        # Capping logic for LLM payload safety on large datasets (>50 terms)
        if total_terms <= 50:
            summary = {
                "result_id": result_id,
                "total_terms": total_terms,
                "threshold": threshold,
                "n_kept": len(kept_terms),
                "n_removed": len(removed_terms),
                "kept_terms": kept_terms,
                "removed_terms": removed_terms,
                "variances": variances_dict
            }
        else:
            summary = {
                "result_id": result_id,
                "total_terms": total_terms,
                "threshold": threshold,
                "n_kept": len(kept_terms),
                "n_removed": len(removed_terms),
                "top_kept_terms": kept_terms[:20],
                "top_removed_terms": removed_terms[:20]
            }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_variances
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        use_abs: bool = False
    ) -> Dict[str, Any]:
        """Filters terms by Pearson correlation with a target class label (one-vs-rest).

        Args:
            target_class: Target category name to correlate terms against.
            threshold: Minimum Pearson correlation coefficient. Terms below this value are filtered.
            use_abs: If True, evaluates absolute correlation |r| >= threshold; otherwise evaluates r >= threshold.
        """
        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            return {"error": "No DTM or dataframe found in session. Call load_dataset_tool and build_dtm_tool first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "No 'category_name' column found in session dataframe."}

        y_binary = (session.dataframe["category_name"] == target_class).astype(float).values
        N = len(y_binary)

        y_std = np.std(y_binary, ddof=0)
        if y_std == 0:
            pearson_rs = np.zeros(len(session.feature_names))
        else:
            # Memory-efficient sparse calculation across all terms
            x_mean = np.array(session.feature_matrix.mean(axis=0)).ravel()
            x_var = np.array(session.feature_matrix.power(2).mean(axis=0)).ravel() - x_mean ** 2
            x_std = np.sqrt(np.maximum(0, x_var))

            xy_mean = np.array(session.feature_matrix.T.dot(y_binary)).ravel() / N
            cov = xy_mean - (x_mean * np.mean(y_binary))

            with np.errstate(divide='ignore', invalid='ignore'):
                pearson_rs = np.where((x_std > 0) & (y_std > 0), cov / (x_std * y_std), 0.0)

        df_pearson = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": pearson_rs
        })

        if use_abs:
            df_pearson["sort_metric"] = df_pearson["pearson_r"].abs()
            df_pearson = df_pearson.sort_values(by="sort_metric", ascending=False).reset_index(drop=True)
            kept_df = df_pearson[df_pearson["sort_metric"] >= threshold]
            removed_df = df_pearson[df_pearson["sort_metric"] < threshold]
            df_pearson = df_pearson.drop(columns=["sort_metric"])
        else:
            df_pearson = df_pearson.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)
            kept_df = df_pearson[df_pearson["pearson_r"] >= threshold]
            removed_df = df_pearson[df_pearson["pearson_r"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        session.artifacts[f"pearson_filtered_terms_{target_class}"] = kept_terms

        total_terms = len(session.feature_names)
        r_dict = dict(zip(df_pearson["term"], [round(float(r), 6) for r in df_pearson["pearson_r"]]))

        result_id = session.next_result_id("pearson_filter")

        if total_terms <= 50:
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "use_abs": use_abs,
                "total_terms": total_terms,
                "n_kept": len(kept_terms),
                "n_removed": len(removed_terms),
                "kept_terms": kept_terms,
                "removed_terms": removed_terms,
                "pearson_r": r_dict
            }
        else:
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "use_abs": use_abs,
                "total_terms": total_terms,
                "n_kept": len(kept_terms),
                "n_removed": len(removed_terms),
                "top_kept_terms": kept_terms[:20],
                "top_removed_terms": removed_terms[:20]
            }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold, "use_abs": use_abs},
            summary=summary,
            full_report=df_pearson
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool]
