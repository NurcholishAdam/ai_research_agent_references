#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference Ingestion Pipeline Demo
Demonstrates the enhanced reference ingestion with semantic graph integration,
citation metadata, and trace-level provenance.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from reference_ingestion import ReferenceIngestionPipeline

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

async def main():
    """Main demonstration function"""
    
    print_section("Reference Ingestion Pipeline Demo")
    print("🔗 Demonstrating enhanced reference ingestion with semantic graph integration")
    print("📊 Features: Citation metadata, provenance tracking, reward shaping")
    
    # Initialize the pipeline
    print_subsection("1. Initialize Pipeline")
    
    try:
        # Try to use semantic graph
        from stage_3_semantic_graph import SemanticGraphManager
        semantic_graph = SemanticGraphManager()
        pipeline = ReferenceIngestionPipeline(semantic_graph=semantic_graph)
        print("✅ Initialized with semantic graph integration")
    except ImportError:
        pipeline = ReferenceIngestionPipeline(storage_path="reference_data")
        print("⚠️ Initialized with fallback NetworkX graph (semantic graph not available)")
    
    # Load references
    print_subsection("2. Load References")
    
    references_file = "references.json"
    if not Path(references_file).exists():
        print(f"❌ References file not found: {references_file}")
        print("Please ensure references.json exists in the current directory")
        return
    
    references = pipeline.load_references(references_file)
    print(f"📚 Loaded {len(references)} references")
    
    # Display sample reference
    if references:
        sample_ref = references[0]
        print(f"\n📄 Sample Reference:")
        print(f"   ID: {sample_ref['id']}")
        print(f"   Title: {sample_ref['title']}")
        print(f"   Authors: {', '.join(sample_ref.get('authors', []))}")
        print(f"   Source: {sample_ref['source']}")
        print(f"   Citations: {sample_ref.get('citations', 0)}")
    
    # Ingest references with full features
    print_subsection("3. Ingest References with Full Features")
    
    results = pipeline.ingest_references(
        references=references,
        extract_citations=True,
        create_provenance=True
    )
    
    print(f"✅ Ingestion Results:")
    print(f"   Nodes created: {len(results['nodes_created'])}")
    print(f"   Citations extracted: {len(results['citations_extracted'])}")
    print(f"   Provenance traces: {len(results['provenance_traces'])}")
    print(f"   Errors: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\n⚠️ Errors encountered:")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    # Demonstrate provenance tracking
    print_subsection("4. Provenance Tracking")
    
    if references:
        ref_id = references[0]['id']
        provenance = pipeline.get_reference_provenance(ref_id)
        
        if provenance:
            print(f"🔍 Provenance for reference '{ref_id}':")
            print(f"   Node ID: {provenance['node_id']}")
            
            if provenance['provenance_trace']:
                trace = provenance['provenance_trace']
                print(f"   Trace ID: {trace['trace_id']}")
                print(f"   Reference Chain: {trace['reference_chain']}")
                print(f"   Confidence Path: {[f'{c:.3f}' for c in trace['confidence_path']]}")
                
                print(f"\n   🎯 Reward Signals:")
                for signal, value in trace['reward_signals'].items():
                    print(f"      {signal}: {value:.3f}")
                
                print(f"\n   📊 Quality Metrics:")
                for metric, value in trace['quality_metrics'].items():
                    print(f"      {metric}: {value:.3f}")
        else:
            print(f"❌ No provenance found for reference '{ref_id}'")
    
    # Demonstrate citation network
    print_subsection("5. Citation Network Analysis")
    
    if references and pipeline.semantic_graph:
        ref_id = references[0]['id']
        network = pipeline.get_citation_network(ref_id, max_depth=2)
        
        if 'error' not in network:
            print(f"🕸️ Citation Network for '{ref_id}':")
            print(f"   Center Reference: {network['center_reference']}")
            print(f"   Connected References: {len(network['neighbors'])}")
            
            if network['neighbors']:
                print(f"\n   📎 Connected References:")
                for neighbor_id, info in list(network['neighbors'].items())[:5]:  # Show first 5
                    print(f"      {neighbor_id}: distance={info['distance']}, "
                          f"relationship={info['relationship']}, "
                          f"confidence={info['confidence']:.3f}")
        else:
            print(f"⚠️ Citation network analysis: {network['error']}")
    else:
        print("⚠️ Citation network analysis requires semantic graph integration")
    
    # Enhance semantic graph with provenance
    print_subsection("6. Semantic Graph Enhancement")
    
    if pipeline.semantic_graph:
        enhancement_stats = pipeline.enhance_semantic_graph_with_provenance()
        print(f"🕸️ Semantic Graph Enhancement:")
        print(f"   Nodes Enhanced: {enhancement_stats['nodes_enhanced']}")
        print(f"   Provenance Links Added: {enhancement_stats['provenance_links_added']}")
        print(f"   Reward Annotations Added: {enhancement_stats['reward_annotations_added']}")
    else:
        print("⚠️ Semantic graph enhancement requires semantic graph integration")
    
    # Show trace quality dashboard
    print_subsection("7. Trace Quality Dashboard")
    
    dashboard = pipeline.get_trace_quality_dashboard()
    
    print(f"📊 Trace Quality Dashboard:")
    print(f"   Total Traces: {dashboard['overview']['total_traces']}")
    print(f"   Total References: {dashboard['overview']['total_references']}")
    
    if dashboard['quality_distribution']:
        quality_dist = dashboard['quality_distribution']
        print(f"\n   🎯 Quality Distribution:")
        print(f"      High Quality (>0.8): {quality_dist['high_quality']}")
        print(f"      Medium Quality (0.6-0.8): {quality_dist['medium_quality']}")
        print(f"      Low Quality (<0.6): {quality_dist['low_quality']}")
    
    if dashboard['reward_signal_analysis']:
        print(f"\n   💰 Top Reward Signals:")
        for signal, stats in list(dashboard['reward_signal_analysis'].items())[:3]:
            print(f"      {signal}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    if dashboard['overall_quality']:
        overall = dashboard['overall_quality']
        print(f"\n   📈 Overall Quality Metrics:")
        print(f"      Average Quality: {overall['average_quality']:.3f}")
        print(f"      High Quality %: {overall['high_quality_percentage']:.1f}%")
    
    # Show comprehensive statistics
    print_subsection("8. Comprehensive Statistics")
    
    stats = pipeline.get_ingestion_statistics()
    
    print(f"📈 Ingestion Statistics:")
    print(f"   References Processed: {stats['references_processed']}")
    print(f"   Citations Extracted: {stats['citations_extracted']}")
    print(f"   Provenance Traces: {stats['provenance_traces_created']}")
    print(f"   Duplicate References Merged: {stats['duplicate_references_merged']}")
    print(f"   Validation Errors: {stats['validation_errors']}")
    
    if 'quality_distribution' in stats:
        quality_dist = stats['quality_distribution']
        print(f"\n   🎯 Quality Distribution:")
        print(f"      High Quality: {quality_dist['high_quality_count']}")
        print(f"      Medium Quality: {quality_dist['medium_quality_count']}")
        print(f"      Low Quality: {quality_dist['low_quality_count']}")
        print(f"      Average Quality: {quality_dist['average_quality']:.3f}")
    
    if 'citation_statistics' in stats:
        citation_stats = stats['citation_statistics']
        print(f"\n   📎 Citation Statistics:")
        print(f"      Total Citations: {citation_stats['total_citations']}")
        
        if citation_stats['citation_types']:
            print(f"      Citation Types:")
            for ctype, count in citation_stats['citation_types'].items():
                print(f"         {ctype}: {count}")
    
    if 'provenance_statistics' in stats:
        prov_stats = stats['provenance_statistics']
        print(f"\n   🔍 Provenance Statistics:")
        print(f"      Total Traces: {prov_stats['total_traces']}")
        print(f"      Average Chain Length: {prov_stats['average_chain_length']:.2f}")
        print(f"      Average Confidence: {prov_stats['average_confidence']:.3f}")
        print(f"      Average Quality Score: {prov_stats['average_quality_score']:.3f}")
    
    # Demonstrate RLHF integration
    print_subsection("9. RLHF Integration Demo")
    
    from reference_ingestion import get_rlhf_reward_signals, calculate_research_quality_score
    
    # Demo RLHF reward signals
    research_context = {
        "query_type": "recent_research",
        "domain": "machine_learning",
        "complexity": "high"
    }
    
    rlhf_rewards = get_rlhf_reward_signals(pipeline, research_context)
    
    print(f"🎯 RLHF Reward Signals:")
    for signal, value in rlhf_rewards.items():
        print(f"   {signal}: {value:.3f}")
    
    # Demo research quality scoring
    if references:
        cited_refs = [ref['id'] for ref in references[:3]]  # Use first 3 as example
        research_output = "This research demonstrates the effectiveness of transformer architectures..."
        
        quality_score = calculate_research_quality_score(pipeline, cited_refs, research_output)
        print(f"\n📊 Research Quality Score: {quality_score:.3f}")
        print(f"   Based on {len(cited_refs)} cited references")
    
    # Demonstrate reward shaping integration
    print_subsection("10. Reward Shaping for AI Research Agent")
    
    print("🎯 Reward Shaping Integration:")
    print("   The reference ingestion pipeline provides rich reward signals that can be used")
    print("   to shape the behavior of the AI Research Agent:")
    print()
    print("   📊 Citation-based rewards: Higher rewards for well-cited references")
    print("   🎯 Completeness rewards: Bonus for references with complete metadata")
    print("   ⏰ Recency rewards: Preference for more recent research")
    print("   🏛️ Source credibility: Different weights for different sources")
    print("   🔗 Network centrality: Rewards for references in dense citation networks")
    print()
    print("   These signals enable the agent to:")
    print("   • Prioritize high-quality sources in research")
    print("   • Learn to identify authoritative references")
    print("   • Develop preferences for complete and recent information")
    print("   • Build better citation networks and knowledge graphs")
    
    # Integration examples
    print_subsection("11. Integration with AI Research Agent Components")
    
    print("🔧 Integration Examples:")
    print()
    print("   Memory System Integration:")
    print("   • Store high-reward references in long-term memory")
    print("   • Use provenance traces for memory retrieval ranking")
    print("   • Apply citation-based importance scoring")
    print()
    print("   Context Engineering:")
    print("   • Pack context with high-quality references first")
    print("   • Use citation networks for context expansion")
    print("   • Apply confidence-based filtering")
    print()
    print("   RLHF Integration:")
    print("   • Use reference quality as reward signal")
    print("   • Train preference models on citation patterns")
    print("   • Optimize for authoritative source selection")
    print()
    print("   Semantic Graph Enhancement:")
    print("   • Enrich graph with citation metadata")
    print("   • Enable provenance-aware retrieval")
    print("   • Support citation-based reasoning")
    
    print_section("Demo Complete")
    print("🎉 Reference ingestion pipeline demonstration finished!")
    print("📚 The pipeline is ready to ingest structured references and provide")
    print("    trace-level provenance and reward shaping for the AI Research Agent.")

if __name__ == "__main__":
    asyncio.run(main())