from typing import List, Optional, Dict, Any
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Peeks at the first few rows of the loaded session DataFrame with truncated text and list previews.

        Args:
            n_rows: Number of rows to inspect from the top of the DataFrame.
            columns: Optional list of column names to inspect. If None, inspects all columns.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first using load_dataset_tool."}

        available_cols = session.dataframe.columns.tolist()

        # Validate requested columns
        if columns is not None:
            missing_cols = [col for col in columns if col not in available_cols]
            if missing_cols:
                return {
                    "error": f"Columns {missing_cols} not found in dataset.",
                    "available_columns": available_cols
                }
            selected_cols = columns
        else:
            selected_cols = available_cols

        # Copy top n_rows slice
        sub_df = session.dataframe[selected_cols].head(n_rows).copy()

        def truncate_val(val: Any) -> Any:
            if isinstance(val, str) and len(val) > 100:
                return val[:100] + "..."
            elif isinstance(val, list):
                if len(val) > 10:
                    truncated_list = [str(item)[:20] for item in val[:10]]
                    truncated_list.append(f"... ({len(val)} items total)")
                    return truncated_list
                else:
                    return [str(item)[:20] for item in val]
            return val

        # Truncate text, string, and list columns to prevent payload bloat
        for col in sub_df.columns:
            sub_df[col] = sub_df[col].apply(truncate_val)

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_rows": len(session.dataframe),
            "total_columns": len(available_cols),
            "available_columns": available_cols,
            "inspected_rows": len(sub_df),
            "data": sub_df.to_dict(orient="records")
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n_rows": n_rows, "columns": columns},
            summary=summary
        )

        return summary

    return [inspect_data_tool]
