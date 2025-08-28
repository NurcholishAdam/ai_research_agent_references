#!/usr/bin/env python3
"""Basic test for reference ingestion pipeline"""

import tempfile
from reference_ingestion import ReferenceIngestionPipeline

def test_basic():
    # Test basic functionality
    temp_dir = tempfile.mkdtemp()
    pipeline = ReferenceIngestionPipeline(storage_path=temp_dir)
    
    # Test with minimal reference
    minimal_ref = {
        'id': 'minimal_test',
        'title': 'Minimal Test Reference',
        'source': 'INTERNAL',
        'type': 'CONCEPT'
    }
    
    results = pipeline.ingest_references([minimal_ref])
    
    print(f'✅ Nodes created: {len(results["nodes_created"])}')
    print(f'✅ Errors: {len(results["errors"])}')
    print(f'✅ References processed: {pipeline.ingestion_stats["references_processed"]}')
    
    assert len(results["nodes_created"]) == 1
    assert len(results["errors"]) == 0
    assert pipeline.ingestion_stats["references_processed"] == 1
    
    print('✅ Basic reference ingestion tests passed!')

if __name__ == "__main__":
    test_basic()