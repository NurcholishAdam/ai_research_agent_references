# -*- coding: utf-8 -*-
"""
Test suite for Reference Ingestion Pipeline
Tests the enhanced reference ingestion with semantic graph integration
"""

import pytest
import networkx as nx
from pathlib import Path
import json
import tempfile
import os

try:
    from reference_ingestion import (
        ReferenceIngestionPipeline, 
        load_references, 
        ingest_references,
        CitationMetadata,
        ProvenanceTrace
    )
except ImportError:
    import sys
    sys.path.append('..')
    from reference_ingestion import (
        ReferenceIngestionPipeline, 
        load_references, 
        ingest_references,
        CitationMetadata,
        ProvenanceTrace
    )

# Test data
SAMPLE_REFERENCES = [
    {
        "id": "test_paper_1",
        "title": "Test Paper on Machine Learning",
        "source": "ARXIV",
        "type": "PAPER",
        "authors": ["John Doe", "Jane Smith"],
        "confidence": 0.9,
        "citations": 25,
        "tags": ["machine learning", "test"],
        "published": "2023-01-15",
        "abstract": "This is a test paper about machine learning concepts.",
        "doi": "10.1000/test.123",
        "venue": "Test Conference",
        "references": ["test_paper_2"]
    },
    {
        "id": "test_paper_2",
        "title": "Foundational Concepts in AI",
        "source": "ARXIV",
        "type": "PAPER",
        "authors": ["Alice Johnson"],
        "confidence": 0.85,
        "citations": 150,
        "tags": ["AI", "foundations"],
        "published": "2022-06-10",
        "abstract": "A comprehensive overview of foundational AI concepts.",
        "doi": "10.1000/test.456"
    },
    {
        "id": "test_code_1",
        "title": "ML Toolkit",
        "source": "GITHUB",
        "type": "CODE",
        "authors": ["Developer Team"],
        "confidence": 0.8,
        "citations": 12,
        "tags": ["toolkit", "python"],
        "published": "2023-03-20",
        "description": "A comprehensive machine learning toolkit.",
        "url": "https://github.com/test/ml-toolkit",
        "language": "Python"
    }
]

class TestReferenceIngestionPipeline:
    """Test the enhanced reference ingestion pipeline"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = ReferenceIngestionPipeline(storage_path=self.temp_dir)
        
        # Create test references file
        self.test_refs_file = os.path.join(self.temp_dir, "test_references.json")
        with open(self.test_refs_file, "w") as f:
            json.dump(SAMPLE_REFERENCES, f)
    
    def test_load_references(self):
        """Test loading references from JSON file"""
        refs = self.pipeline.load_references(self.test_refs_file)
        
        assert len(refs) == 3
        assert refs[0]["id"] == "test_paper_1"
        assert refs[1]["id"] == "test_paper_2"
        assert refs[2]["id"] == "test_code_1"
    
    def test_reference_validation(self):
        """Test reference structure validation"""
        # Valid reference
        valid_ref = {"id": "valid_123", "title": "Valid Paper"}
        assert self.pipeline._validate_reference_structure(valid_ref)
        
        # Invalid reference (missing id)
        invalid_ref = {"title": "Invalid Paper"}
        assert not self.pipeline._validate_reference_structure(invalid_ref)
    
    def test_map_reference_to_node_data(self):
        """Test mapping reference to node data"""
        ref = SAMPLE_REFERENCES[0]
        node_data = self.pipeline._map_reference_to_node_data(ref)
        
        assert "content" in node_data
        assert "title" in node_data
        assert "node_type" in node_data
        assert "source_type" in node_data
        assert "metadata" in node_data
        assert "tags" in node_data
        assert "importance_score" in node_data
        assert "confidence_score" in node_data
        
        # Check content includes title and abstract
        assert "Test Paper on Machine Learning" in node_data["content"]
        assert "This is a test paper" in node_data["content"]
        
        # Check metadata
        assert node_data["metadata"]["authors"] == ["John Doe", "Jane Smith"]
        assert node_data["metadata"]["citation_count"] == 25
        assert node_data["metadata"]["doi"] == "10.1000/test.123"
    
    def test_ingest_references_basic(self):
        """Test basic reference ingestion"""
        refs = self.pipeline.load_references(self.test_refs_file)
        results = self.pipeline.ingest_references(refs, extract_citations=False, create_provenance=False)
        
        assert len(results["nodes_created"]) == 3
        assert len(results["errors"]) == 0
        assert self.pipeline.ingestion_stats["references_processed"] == 3
        
        # Check reference mappings
        assert "test_paper_1" in self.pipeline.reference_mappings
        assert "test_paper_2" in self.pipeline.reference_mappings
        assert "test_code_1" in self.pipeline.reference_mappings
    
    def test_ingest_references_with_citations(self):
        """Test reference ingestion with citation extraction"""
        refs = self.pipeline.load_references(self.test_refs_file)
        results = self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=False)
        
        assert len(results["nodes_created"]) == 3
        assert len(results["citations_extracted"]) > 0  # Should extract author relationships
        assert self.pipeline.ingestion_stats["citations_extracted"] > 0
    
    def test_ingest_references_with_provenance(self):
        """Test reference ingestion with provenance creation"""
        refs = self.pipeline.load_references(self.test_refs_file)
        results = self.pipeline.ingest_references(refs, extract_citations=False, create_provenance=True)
        
        assert len(results["nodes_created"]) == 3
        assert len(results["provenance_traces"]) == 3
        assert self.pipeline.ingestion_stats["provenance_traces_created"] == 3
        
        # Check provenance traces
        assert len(self.pipeline.provenance_traces) == 3
        for trace in self.pipeline.provenance_traces.values():
            assert isinstance(trace, ProvenanceTrace)
            assert len(trace.reference_chain) > 0
            assert len(trace.reasoning_steps) > 0
            assert "citation_reward" in trace.reward_signals
            assert "completeness_reward" in trace.reward_signals
    
    def test_citation_creation(self):
        """Test citation relationship creation"""
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=False, create_provenance=False)
        
        # Create citation relationship
        citation_id = self.pipeline._create_citation_relationship(
            source_ref="test_paper_1",
            target_ref="test_paper_2",
            citation_type="direct",
            confidence=0.9
        )
        
        assert citation_id is not None
        assert citation_id in self.pipeline.citations
        
        citation = self.pipeline.citations[citation_id]
        assert citation.source_reference == "test_paper_1"
        assert citation.target_reference == "test_paper_2"
        assert citation.citation_type == "direct"
        assert citation.confidence_score == 0.9
    
    def test_provenance_trace_creation(self):
        """Test provenance trace creation"""
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=False, create_provenance=False)
        
        ref = SAMPLE_REFERENCES[0]
        trace = self.pipeline._create_provenance_trace(ref)
        
        assert trace is not None
        assert isinstance(trace, ProvenanceTrace)
        assert trace.reference_chain == ["test_paper_1"]
        assert len(trace.reasoning_steps) > 0
        assert "citation_reward" in trace.reward_signals
        assert "metadata_completeness" in trace.quality_metrics
    
    def test_reward_signal_calculation(self):
        """Test reward signal calculation"""
        ref = SAMPLE_REFERENCES[0]
        signals = self.pipeline._calculate_reference_reward_signals(ref)
        
        assert "citation_reward" in signals
        assert "completeness_reward" in signals
        assert "recency_reward" in signals
        assert "source_credibility" in signals
        
        # Check values are in valid range
        for signal_value in signals.values():
            assert 0 <= signal_value <= 1
        
        # Check completeness reward (should be high for complete reference)
        assert signals["completeness_reward"] == 1.0  # All required fields present
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        ref = SAMPLE_REFERENCES[0]
        metrics = self.pipeline._calculate_reference_quality_metrics(ref)
        
        assert "metadata_completeness" in metrics
        assert "content_richness" in metrics
        assert "authority_score" in metrics
        
        # Check values are in valid range
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1
    
    def test_get_reference_provenance(self):
        """Test getting reference provenance information"""
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=True)
        
        provenance = self.pipeline.get_reference_provenance("test_paper_1")
        
        assert provenance is not None
        assert provenance["reference_id"] == "test_paper_1"
        assert "node_id" in provenance
        assert "provenance_trace" in provenance
        assert "related_citations" in provenance
        assert "quality_metrics" in provenance
        assert "reward_signals" in provenance
    
    def test_get_ingestion_statistics(self):
        """Test getting ingestion statistics"""
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=True)
        
        stats = self.pipeline.get_ingestion_statistics()
        
        assert "references_processed" in stats
        assert "citations_extracted" in stats
        assert "provenance_traces_created" in stats
        assert "citation_statistics" in stats
        assert "provenance_statistics" in stats
        assert "quality_distribution" in stats
        
        assert stats["references_processed"] == 3
        assert stats["provenance_traces_created"] == 3
        
        # Test quality distribution
        quality_dist = stats["quality_distribution"]
        assert "high_quality_count" in quality_dist
        assert "average_quality" in quality_dist
    
    def test_enhanced_reward_signals(self):
        """Test enhanced reward signal calculation"""
        ref = SAMPLE_REFERENCES[0]
        signals = self.pipeline._calculate_reference_reward_signals(ref)
        
        # Check all enhanced signals are present
        expected_signals = [
            "citation_reward", "completeness_reward", "recency_reward",
            "source_credibility", "network_centrality", "author_authority",
            "content_richness", "cross_reference_reward"
        ]
        
        for signal in expected_signals:
            assert signal in signals
            assert 0 <= signals[signal] <= 1
    
    def test_composite_quality_score(self):
        """Test composite quality score calculation"""
        ref = SAMPLE_REFERENCES[0]
        reward_signals = self.pipeline._calculate_reference_reward_signals(ref)
        quality_metrics = self.pipeline._calculate_reference_quality_metrics(ref)
        
        composite_score = self.pipeline._calculate_composite_quality_score(
            reward_signals, quality_metrics
        )
        
        assert 0 <= composite_score <= 1
        assert isinstance(composite_score, float)
    
    def test_trace_quality_dashboard(self):
        """Test trace quality dashboard generation"""
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=True)
        
        dashboard = self.pipeline.get_trace_quality_dashboard()
        
        assert "overview" in dashboard
        assert "quality_distribution" in dashboard
        assert "reward_signal_analysis" in dashboard
        
        # Check overview
        overview = dashboard["overview"]
        assert overview["total_traces"] == 3
        assert overview["total_references"] == 3
    
    def test_semantic_graph_enhancement(self):
        """Test semantic graph enhancement with provenance"""
        if not self.has_semantic_graph:
            pytest.skip("Semantic graph not available")
        
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=True)
        
        enhancement_stats = self.pipeline.enhance_semantic_graph_with_provenance()
        
        assert "nodes_enhanced" in enhancement_stats
        assert "provenance_links_added" in enhancement_stats
        assert "reward_annotations_added" in enhancement_stats
        
        assert enhancement_stats["nodes_enhanced"] > 0
    
    def test_rlhf_integration(self):
        """Test RLHF reward signal generation"""
        from reference_ingestion import get_rlhf_reward_signals, calculate_research_quality_score
        
        refs = self.pipeline.load_references(self.test_refs_file)
        self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=True)
        
        # Test RLHF reward signals
        research_context = {"query_type": "recent_research"}
        rlhf_rewards = get_rlhf_reward_signals(self.pipeline, research_context)
        
        expected_rewards = [
            "reference_quality_reward", "citation_authority_reward",
            "source_diversity_reward", "temporal_relevance_reward",
            "network_connectivity_reward"
        ]
        
        for reward in expected_rewards:
            assert reward in rlhf_rewards
            assert 0 <= rlhf_rewards[reward] <= 1
        
        # Test research quality scoring
        cited_refs = ["test_paper_1", "test_paper_2"]
        research_output = "Test research output"
        
        quality_score = calculate_research_quality_score(
            self.pipeline, cited_refs, research_output
        )
        
        assert 0 <= quality_score <= 1
    
    def test_backward_compatibility(self):
        """Test backward compatibility functions"""
        # Test load_references function
        refs = load_references(self.test_refs_file)
        assert len(refs) == 3
        
        # Test NetworkX graph ingestion
        G = nx.MultiDiGraph()
        ingest_references(G, refs)
        assert len(G.nodes) == len(refs)
        
        for node_id, data in G.nodes(data=True):
            assert "source_id" in data
            assert "confidence_score" in data
            assert isinstance(data["tags"], list)

class TestIntegrationWithSemanticGraph:
    """Test integration with semantic graph (if available)"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Try to create pipeline with semantic graph
        try:
            from stage_3_semantic_graph import SemanticGraphManager
            semantic_graph = SemanticGraphManager()
            self.pipeline = ReferenceIngestionPipeline(
                semantic_graph=semantic_graph,
                storage_path=self.temp_dir
            )
            self.has_semantic_graph = True
        except ImportError:
            self.pipeline = ReferenceIngestionPipeline(storage_path=self.temp_dir)
            self.has_semantic_graph = False
        
        # Create test references file
        self.test_refs_file = os.path.join(self.temp_dir, "test_references.json")
        with open(self.test_refs_file, "w") as f:
            json.dump(SAMPLE_REFERENCES, f)
    
    def test_semantic_graph_integration(self):
        """Test integration with semantic graph"""
        if not self.has_semantic_graph:
            pytest.skip("Semantic graph not available")
        
        refs = self.pipeline.load_references(self.test_refs_file)
        results = self.pipeline.ingest_references(refs, extract_citations=True, create_provenance=True)
        
        assert len(results["nodes_created"]) == 3
        assert len(results["citations_extracted"]) > 0
        
        # Check that nodes were added to semantic graph
        assert len(self.pipeline.semantic_graph.nodes) >= 3
        
        # Check citation network
        network = self.pipeline.get_citation_network("test_paper_1")
        assert "center_reference" in network
        assert network["center_reference"] == "test_paper_1"

def test_basic_functionality():
    """Basic functionality test for CI/CD"""
    # Create temporary pipeline
    temp_dir = tempfile.mkdtemp()
    pipeline = ReferenceIngestionPipeline(storage_path=temp_dir)
    
    # Test with minimal reference
    minimal_ref = {
        "id": "minimal_test",
        "title": "Minimal Test Reference",
        "source": "INTERNAL",
        "type": "CONCEPT"
    }
    
    results = pipeline.ingest_references([minimal_ref])
    
    assert len(results["nodes_created"]) == 1
    assert len(results["errors"]) == 0
    assert pipeline.ingestion_stats["references_processed"] == 1

if __name__ == "__main__":
    # Run basic tests
    test_basic_functionality()
    print("✅ Basic reference ingestion tests passed!")
    
    # Run full test suite if pytest is available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Install pytest for full test suite: pip install pytest")
