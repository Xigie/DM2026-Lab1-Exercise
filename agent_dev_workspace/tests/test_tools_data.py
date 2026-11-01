import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_inspect_data_tool_normal():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    result = inspect_data_tool.invoke({"n_rows": 3})
    
    assert result["total_rows"] == 8
    assert result["inspected_rows"] == 3
    assert len(result["data"]) == 3
    assert result["data"][0]["category_name"] == "catA"
    assert result["data"][0]["text"] == "always alpha alpha alpha beta"  # Short text not truncated

def test_inspect_data_tool_no_data():
    session = SessionState()
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    result = inspect_data_tool.invoke({})
    assert "error" in result
    assert "No dataset loaded" in result["error"]

def test_inspect_data_tool_missing_column():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    result = inspect_data_tool.invoke({"columns": ["text", "non_existent_col"]})
    assert "error" in result
    assert "non_existent_col" in result["error"]
    assert "available_columns" in result

def test_inspect_data_tool_truncation():
    session = SessionState()
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    # Create dummy DataFrame with long text and long list column
    long_text = "word " * 30  # 150 chars
    long_list = [f"tok_{i}" for i in range(25)]
    
    session.dataframe = pd.DataFrame({
        "text": [long_text],
        "unigrams": [long_list]
    })
    
    result = inspect_data_tool.invoke({"n_rows": 1})
    inspected_row = result["data"][0]
    
    # Check text truncation (>100 chars -> 100 + "...")
    assert inspected_row["text"].endswith("...")
    assert len(inspected_row["text"]) == 103
    
    # Check list truncation (>10 items -> 10 items + summary string)
    assert len(inspected_row["unigrams"]) == 11
    assert inspected_row["unigrams"][-1] == "... (25 items total)"
