def test_dashboard_metrics_extraction():
    from trace_quality_dashboard import get_trace_metrics
    df = get_trace_metrics()
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ["confidence", "reward", "novelty", "centrality"])
