#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SSRL-Integrated Trace Buffer System
============================================

Comprehensive tests for the SSRL-integrated trace buffer including:
- Basic trace operations
- SSRL integration
- Sampling strategies
- Quality-based operations
- Buffer management
"""

import pytest
import asyncio
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any

from extensions.stage_8_trace_buffer import (
    SSRLTraceBuffer, TraceType, SamplingStrategy, BufferPriority,
    TraceMetadata, TraceEntry, integrate_ssrl_trace_buffer
)

try:
    from extensions.stage_9_ssrl import SSRLConfig, ModalityType
    SSRL_AVAILABLE = True
except ImportError:
    SSRL_AVAILABLE = False

class TestSSRLTraceBuffer:
    """Test suite for SSRL trace buffer"""
    
    @pytest.fixture
    def trace_buffer(self):
        """Create a trace buffer for testing"""
        return SSRLTraceBuffer(max_size=10)
    
    @pytest.fixture
    def sample_trace_data(self):
        """Sample trace data for testing"""
        return {
            "input_data": {"query": "How do neural networks work?"},
            "output_data": {"response": "Neural networks are computational models..."},
            "trace_type": TraceType.RESEARCH_QUERY,
            "session_id": "test_session",
            "success": True,
            "confidence_score": 0.8,
            "priority": BufferPriority.MEDIUM
        }
    
    @pytest.mark.asyncio
    async def test_basic_trace_operations(self, trace_buffer, sample_trace_data):
        """Test basic trace add and retrieve operations"""
        
        # Add trace
        trace_id = await trace_buffer.add_trace(**sample_trace_data)
        
        assert trace_id is not None
        assert len(trace_buffer.buffer) == 1
        assert trace_id in trace_buffer.trace_index
        
        # Retrieve trace
        retrieved_trace = trace_buffer.get_trace_by_id(trace_id)
        assert retrieved_trace is not None
        assert retrieved_trace.metadata.trace_id == trace_id
        assert retrieved_trace.metadata.success == True
        assert retrieved_trace.metadata.confidence_score == 0.8
    
    @pytest.mark.asyncio
    async def test_buffer_capacity_management(self, sample_trace_data):
        """Test buffer capacity and eviction"""
        
        # Create small buffer
        trace_buffer = SSRLTraceBuffer(max_size=3)
        
        # Add traces beyond capacity
        trace_ids = []
        for i in range(5):
            trace_data = sample_trace_data.copy()
            trace_data["session_id"] = f"session_{i}"
            trace_data["priority"] = BufferPriority.LOW if i < 2 else BufferPriority.HIGH
            
            trace_id = await trace_buffer.add_trace(**trace_data)
            trace_ids.append(trace_id)
        
        # Buffer should not exceed max size
        assert len(trace_buffer.buffer) == 3
        
        # Low priority traces should be evicted first
        assert trace_buffer.get_trace_by_id(trace_ids[0]) is None  # First low priority evicted
        assert trace_buffer.get_trace_by_id(trace_ids[1]) is None  # Second low priority evicted
        assert trace_buffer.get_trace_by_id(trace_ids[2]) is not None  # High priority kept
    
    def test_sampling_strategies(self, trace_buffer):
        """Test different sampling strategies"""
        
        # Add traces with different characteristics
        traces_data = [
            {"success": True, "confidence_score": 0.9, "quality_score": 0.8},
            {"success": False, "confidence_score": 0.3, "quality_score": 0.2},
            {"success": True, "confidence_score": 0.7, "quality_score": 0.6},
            {"success": False, "confidence_score": 0.4, "quality_score": 0.3}
        ]
        
        # Manually add traces to buffer for testing
        for i, trace_data in enumerate(traces_data):
            metadata = TraceMetadata(
                trace_id=f"test_trace_{i}",
                trace_type=TraceType.RESEARCH_QUERY,
                timestamp=datetime.now(),
                session_id="test_session",
                success=trace_data["success"],
                confidence_score=trace_data["confidence_score"],
                quality_score=trace_data["quality_score"]
            )
            
            trace_entry = TraceEntry(
                metadata=metadata,
                input_data={"query": f"test query {i}"},
                output_data={"response": f"test response {i}"}
            )
            
            trace_buffer.buffer.append(trace_entry)
            trace_buffer.trace_index[metadata.trace_id] = len(trace_buffer.buffer) - 1
        
        # Test success-based sampling
        success_batch = trace_buffer.sample_replay_batch(
            batch_size=2, 
            strategy=SamplingStrategy.SUCCESS_BASED
        )
        assert all(trace.metadata.success for trace in success_batch)
        
        # Test confidence-based sampling
        confidence_batch = trace_buffer.sample_replay_batch(
            batch_size=2,
            strategy=SamplingStrategy.CONFIDENCE_BASED,
            confidence_threshold=0.6
        )
        assert all(trace.metadata.confidence_score >= 0.6 for trace in confidence_batch)
        
        # Test quality-based sampling
        quality_batch = trace_buffer.sample_replay_batch(
            batch_size=2,
            strategy=SamplingStrategy.QUALITY_BASED
        )
        # Should return higher quality traces
        avg_quality = np.mean([trace.metadata.quality_score for trace in quality_batch])
        assert avg_quality > 0.5
    
    def test_trace_filtering(self, trace_buffer):
        """Test trace filtering functionality"""
        
        # Add traces with different characteristics
        traces_data = [
            {"trace_type": TraceType.RESEARCH_QUERY, "success": True, "confidence_score": 0.9},
            {"trace_type": TraceType.CODE_GENERATION, "success": False, "confidence_score": 0.3},
            {"trace_type": TraceType.RESEARCH_QUERY, "success": True, "confidence_score": 0.7},
            {"trace_type": TraceType.REASONING_STEP, "success": True, "confidence_score": 0.8}
        ]
        
        # Manually add traces
        for i, trace_data in enumerate(traces_data):
            metadata = TraceMetadata(
                trace_id=f"filter_test_{i}",
                trace_type=trace_data["trace_type"],
                timestamp=datetime.now(),
                session_id="filter_session",
                success=trace_data["success"],
                confidence_score=trace_data["confidence_score"]
            )
            
            trace_entry = TraceEntry(
                metadata=metadata,
                input_data={"test": f"data_{i}"},
                output_data={"result": f"output_{i}"}
            )
            
            trace_buffer.buffer.append(trace_entry)
            trace_buffer.trace_index[metadata.trace_id] = len(trace_buffer.buffer) - 1
        
        # Test filtering by trace type
        research_traces = trace_buffer.filter_traces(trace_type=TraceType.RESEARCH_QUERY)
        assert len(research_traces) == 2
        assert all(trace.metadata.trace_type == TraceType.RESEARCH_QUERY for trace in research_traces)
        
        # Test filtering by success
        successful_traces = trace_buffer.filter_traces(success=True)
        assert len(successful_traces) == 3
        assert all(trace.metadata.success for trace in successful_traces)
        
        # Test filtering by minimum confidence
        high_confidence_traces = trace_buffer.filter_traces(min_confidence=0.75)
        assert len(high_confidence_traces) == 2
        assert all(trace.metadata.confidence_score >= 0.75 for trace in high_confidence_traces)
        
        # Test combined filtering
        combined_filter = trace_buffer.filter_traces(
            trace_type=TraceType.RESEARCH_QUERY,
            success=True,
            min_confidence=0.8
        )
        assert len(combined_filter) == 1
        assert combined_filter[0].metadata.confidence_score == 0.9
    
    def test_buffer_statistics(self, trace_buffer):
        """Test buffer statistics calculation"""
        
        # Add sample traces
        for i in range(5):
            metadata = TraceMetadata(
                trace_id=f"stats_test_{i}",
                trace_type=TraceType.RESEARCH_QUERY,
                timestamp=datetime.now(),
                session_id="stats_session",
                success=i % 2 == 0,  # Alternating success
                confidence_score=0.5 + (i * 0.1),
                quality_score=0.4 + (i * 0.1)
            )
            
            trace_entry = TraceEntry(
                metadata=metadata,
                input_data={"test": f"data_{i}"},
                output_data={"result": f"output_{i}"}
            )
            
            trace_buffer.buffer.append(trace_entry)
            trace_buffer.trace_index[metadata.trace_id] = len(trace_buffer.buffer) - 1
        
        # Get statistics
        stats = trace_buffer.get_buffer_statistics()
        
        assert stats["current_size"] == 5
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.5
        assert stats["success_rate"] == 0.6  # 3 out of 5 successful
        assert 0.6 <= stats["avg_confidence_score"] <= 0.7
        assert 0.5 <= stats["avg_quality_score"] <= 0.6
    
    @pytest.mark.skipif(not SSRL_AVAILABLE, reason="SSRL components not available")
    @pytest.mark.asyncio
    async def test_ssrl_integration(self, sample_trace_data):
        """Test SSRL integration functionality"""
        
        # Create trace buffer with SSRL
        ssrl_config = SSRLConfig(encoder_dim=256, projection_dim=128)
        trace_buffer = SSRLTraceBuffer(max_size=10, ssrl_config=ssrl_config)
        
        # Add trace with SSRL enhancement
        trace_id = await trace_buffer.add_trace(**sample_trace_data)
        
        # Retrieve trace and check SSRL enhancement
        trace = trace_buffer.get_trace_by_id(trace_id)
        assert trace is not None
        
        # Check if SSRL enhancement was applied
        if trace_buffer.ssrl_system:
            assert hasattr(trace.metadata, 'representation')
            assert hasattr(trace.metadata, 'modality')
            # Note: representation might be None if SSRL enhancement failed
    
    def test_save_and_load_buffer(self, trace_buffer, sample_trace_data, tmp_path):
        """Test buffer save and load functionality"""
        
        # Add sample traces
        trace_ids = []
        for i in range(3):
            trace_data = sample_trace_data.copy()
            trace_data["session_id"] = f"save_test_{i}"
            
            # Create trace entry manually for testing
            metadata = TraceMetadata(
                trace_id=f"save_test_trace_{i}",
                trace_type=trace_data["trace_type"],
                timestamp=datetime.now(),
                session_id=trace_data["session_id"],
                success=trace_data["success"],
                confidence_score=trace_data["confidence_score"],
                priority=trace_data["priority"]
            )
            
            trace_entry = TraceEntry(
                metadata=metadata,
                input_data=trace_data["input_data"],
                output_data=trace_data["output_data"]
            )
            
            trace_buffer.buffer.append(trace_entry)
            trace_buffer.trace_index[metadata.trace_id] = len(trace_buffer.buffer) - 1
            trace_ids.append(metadata.trace_id)
        
        # Save buffer
        save_path = tmp_path / "test_buffer.json"
        trace_buffer.save_buffer(str(save_path))
        
        assert save_path.exists()
        
        # Create new buffer and load
        new_buffer = SSRLTraceBuffer(max_size=10)
        new_buffer.load_buffer(str(save_path))
        
        # Verify loaded data
        assert len(new_buffer.buffer) == 3
        for trace_id in trace_ids:
            loaded_trace = new_buffer.get_trace_by_id(trace_id)
            assert loaded_trace is not None
            assert loaded_trace.metadata.trace_id == trace_id

class TestIntegrationFunction:
    """Test the integration function"""
    
    def test_integrate_ssrl_trace_buffer(self):
        """Test the integration function"""
        
        trace_buffer = integrate_ssrl_trace_buffer(max_size=100)
        
        assert trace_buffer is not None
        assert trace_buffer.max_size == 100
        assert len(trace_buffer.buffer) == 0
        assert isinstance(trace_buffer, SSRLTraceBuffer)
    
    @pytest.mark.skipif(not SSRL_AVAILABLE, reason="SSRL components not available")
    def test_integrate_with_ssrl_config(self):
        """Test integration with SSRL config"""
        
        ssrl_config = SSRLConfig(encoder_dim=512, projection_dim=256)
        trace_buffer = integrate_ssrl_trace_buffer(max_size=50, ssrl_config=ssrl_config)
        
        assert trace_buffer is not None
        assert trace_buffer.max_size == 50
        assert trace_buffer.ssrl_system is not None

# Async test runner for pytest
def test_trace_add_and_sample():
    """Legacy test for compatibility"""
    
    async def async_test():
        tb = SSRLTraceBuffer(max_size=3)
        
        # Add traces
        trace_ids = []
        for i, (success, confidence) in enumerate([(True, 0.9), (False, 0.5), (True, 0.8), (True, 0.95)]):
            trace_id = await tb.add_trace(
                input_data={"query": f"test query {i}"},
                output_data={"response": f"test response {i}"},
                trace_type=TraceType.RESEARCH_QUERY,
                session_id=f"test_session_{i}",
                success=success,
                confidence_score=confidence
            )
            trace_ids.append(trace_id)
        
        # Buffer should not exceed max size
        assert len(tb.buffer) == 3
        
        # Test success-based sampling
        batch = tb.sample_replay_batch(batch_size=2, strategy=SamplingStrategy.SUCCESS_BASED)
        assert all(t.metadata.success for t in batch)
        
        return tb
    
    # Run async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_test())
        assert result is not None
    finally:
        loop.close()

if __name__ == "__main__":
    # Run basic tests
    test_trace_add_and_sample()
    print("✅ Basic trace buffer tests passed!")
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("⚠️ pytest not available, skipping advanced tests")
