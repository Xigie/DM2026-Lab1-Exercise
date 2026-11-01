import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_known_answer_5():
    session = SessionState()
    
    # 1. Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 2. Instantiate student tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    # 3. Load dataset & build DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # 4. Invoke variance_filter_tool with threshold=0.0
    result = variance_filter_tool.invoke({"threshold": 0.0})
    
    # Check exact variance numbers from Known Answer #5
    variances = result["variances"]
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)
    
    # Check full_report format for visualize_result_tool contract
    result_id = result["result_id"]
    full_report = session.results_store[result_id]["full_report"]
    assert list(full_report.columns) == ["term", "variance"]
    assert full_report.iloc[0]["term"] == "alpha"
    assert full_report.iloc[-1]["term"] == "always"

def test_variance_filter_tool_threshold_0_15():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # Invoke with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})
    
    # Assert threshold=0.15 keeps alpha, beta, delta, gamma and removes ONLY always
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms"] == ["always"]

def test_pearson_filter_tool_known_answer_6():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # Invoke pearson_filter_tool for target_class="catB"
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": -1.0, "use_abs": False})
    
    # Assert numbers from Known Answer #6 in TEST_FIXTURE.md
    rs = result["pearson_r"]
    assert rs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert rs["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert rs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert rs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert rs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    
    # Check full_report format for visualize_result_tool
    result_id = result["result_id"]
    full_report = session.results_store[result_id]["full_report"]
    assert list(full_report.columns) == ["term", "pearson_r"]
    assert full_report.iloc[0]["term"] == "delta"

def test_pearson_filter_tool_use_abs():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # Invoke with use_abs=True
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0, "use_abs": True})
    
    kept = result["kept_terms"]
    # Check beta (|r|=0.7746) ranks above always (|r|=0.3780)
    assert kept.index("beta") < kept.index("always")
