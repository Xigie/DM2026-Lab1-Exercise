import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool():
    # 1. Initialize session and tools
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    
    # 2. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 3. Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})
    
    # 4. Assert against Known Answer #4 from TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    
    # Check vocabulary term names stored in session
    expected_vocabulary = ["alpha", "always", "beta", "delta", "gamma"]
    assert sorted(session.feature_names) == expected_vocabulary
    
    # Check count_vectorizer artifact is present
    assert "count_vectorizer" in session.artifacts
    assert session.feature_matrix.shape == (8, 5)

def test_term_frequency_tool():
    # 1. Initialize session and tools
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")
    
    # 2. Load dataset fixture & build DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # 3. Invoke term_frequency_tool
    result = term_frequency_tool.invoke({})
    
    # 4. Assert against Known Answer #14 from TEST_FIXTURE.md
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert result["total_terms"] == 5
    assert result["term_frequencies"] == expected_counts

def test_dtm_heatmap_tool():
    # 1. Initialize session and tools
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")
    
    # 2. Load dataset fixture & build DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # 3. Invoke dtm_heatmap_tool (default n_terms=20, n_documents=20)
    result = dtm_heatmap_tool.invoke({})
    
    # 4. Assert against Known Answer #21 full matrix in TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7
    ]
    assert result["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None
    
    # 5. Check corner slice (n_terms=2, n_documents=3)
    corner_result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert corner_result["n_documents"] == 3
    assert corner_result["n_terms"] == 2
    assert corner_result["terms"] == ["alpha", "always"]
    assert corner_result["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
