#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Reference Integration Example
Demonstrates advanced integration of the reference ingestion pipeline with
the AI Research Agent for trace-level provenance and reward shaping.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the enhanced reference ingestion pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference_ingestion import (
    ReferenceIngestionPipeline, 
    get_rlhf_reward_signals,
    calculate_research_quality_score
)

class EnhancedResearchAgent:
    """
    Example AI Research Agent with enhanced reference integration
    """
    
    def __init__(self):
        # Initialize reference ingestion pipeline
        try:
            from stage_3_semantic_graph import SemanticGraphManager
            self.semantic_graph = SemanticGraphManager()
            self.reference_pipeline = ReferenceIngestionPipeline(
                semantic_graph=self.semantic_graph
            )
            print("✅ Initialized with semantic graph integration")
        except ImportError:
            self.reference_pipeline = ReferenceIngestionPipeline()
            self.semantic_graph = None
            print("⚠️ Initialized without semantic graph (fallback mode)")
        
        # Research session state
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.research_history = []
        self.quality_metrics = []
        
        print(f"🤖 Enhanced Research Agent initialized (Session: {self.session_id})")
    
    async def load_knowledge_base(self, references_file: str = "references.json"):
        """Load and ingest references into the knowledge base"""
        
        print(f"\n📚 Loading knowledge base from {references_file}")
        
        if not Path(references_file).exists():
            print(f"❌ References file not found: {references_file}")
            return False
        
        # Load references
        references = self.reference_pipeline.load_references(references_file)
        print(f"📖 Loaded {len(references)} references")
        
        # Ingest with full features
        results = self.reference_pipeline.ingest_references(
            references=references,
            extract_citations=True,
            create_provenance=True
        )
        
        print(f"✅ Knowledge base ingestion complete:")
        print(f"   Nodes created: {len(results['nodes_created'])}")
        print(f"   Citations extracted: {len(results['citations_extracted'])}")
        print(f"   Provenance traces: {len(results['provenance_traces'])}")
        
        # Enhance semantic graph with provenance
        if self.semantic_graph:
            enhancement_stats = self.reference_pipeline.enhance_semantic_graph_with_provenance()
            print(f"   Semantic graph enhanced: {enhancement_stats['nodes_enhanced']} nodes")
        
        return True
    
    async def conduct_research(self, query: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Conduct research with enhanced reference integration
        
        Args:
            query: Research query
            query_type: Type of query (recent_research, foundational_concepts, etc.)
            
        Returns:
            Research results with quality metrics and provenance
        """
        
        print(f"\n🔍 Conducting research: '{query}'")
        print(f"   Query type: {query_type}")
        
        research_start_time = datetime.now()
        
        # Step 1: Retrieve relevant references
        relevant_refs = await self._retrieve_relevant_references(query, query_type)
        
        # Step 2: Generate research response
        research_response = await self._generate_research_response(query, relevant_refs)
        
        # Step 3: Calculate quality metrics
        quality_metrics = await self._calculate_research_quality(
            query, relevant_refs, research_response, query_type
        )
        
        # Step 4: Generate RLHF reward signals
        rlhf_rewards = self._generate_rlhf_rewards(query_type)
        
        # Step 5: Create research record
        research_record = {
            "session_id": self.session_id,
            "timestamp": research_start_time.isoformat(),
            "query": query,
            "query_type": query_type,
            "relevant_references": relevant_refs,
            "research_response": research_response,
            "quality_metrics": quality_metrics,
            "rlhf_rewards": rlhf_rewards,
            "provenance_traces": self._get_provenance_for_references(relevant_refs)
        }
        
        # Store in research history
        self.research_history.append(research_record)
        self.quality_metrics.append(quality_metrics["overall_quality"])
        
        print(f"✅ Research completed in {(datetime.now() - research_start_time).total_seconds():.2f}s")
        print(f"   Quality Score: {quality_metrics['overall_quality']:.3f}")
        print(f"   References Used: {len(relevant_refs)}")
        
        return research_record
    
    async def _retrieve_relevant_references(self, query: str, query_type: str) -> List[str]:
        """Retrieve relevant references based on query"""
        
        # Simple keyword-based retrieval (in practice, use semantic search)
        query_words = set(query.lower().split())
        
        relevant_refs = []
        
        # Get all reference IDs and their provenance
        for ref_id in self.reference_pipeline.reference_mappings.keys():
            provenance = self.reference_pipeline.get_reference_provenance(ref_id)
            
            if provenance and provenance.get("provenance_trace"):
                trace = provenance["provenance_trace"]
                
                # Check if reference is relevant (simple keyword matching)
                content = " ".join(trace["reasoning_steps"]).lower()
                
                # Calculate relevance score
                relevance_score = 0
                for word in query_words:
                    if word in content:
                        relevance_score += 1
                
                # Include if relevant and high quality
                composite_quality = trace["quality_metrics"].get("composite_quality", 0.5)
                
                if relevance_score > 0 and composite_quality > 0.4:
                    relevant_refs.append(ref_id)
        
        # Sort by quality and return top references
        def get_quality(ref_id):
            prov = self.reference_pipeline.get_reference_provenance(ref_id)
            if prov and prov.get("provenance_trace"):
                return prov["provenance_trace"]["quality_metrics"].get("composite_quality", 0.5)
            return 0.5
        
        relevant_refs.sort(key=get_quality, reverse=True)
        
        # Return top 5 references
        return relevant_refs[:5]
    
    async def _generate_research_response(self, query: str, relevant_refs: List[str]) -> str:
        """Generate research response based on relevant references"""
        
        # In a real implementation, this would use an LLM
        # For demo purposes, we'll create a structured response
        
        response_parts = [
            f"Research Analysis: {query}",
            "",
            "Based on the analysis of relevant literature, here are the key findings:",
            ""
        ]
        
        for i, ref_id in enumerate(relevant_refs, 1):
            provenance = self.reference_pipeline.get_reference_provenance(ref_id)
            
            if provenance and provenance.get("provenance_trace"):
                trace = provenance["provenance_trace"]
                quality = trace["quality_metrics"].get("composite_quality", 0.5)
                
                # Extract title from reasoning steps
                title = "Unknown Reference"
                for step in trace["reasoning_steps"]:
                    if "Reference" in step and "ingested" in step:
                        title = f"Reference {ref_id}"
                        break
                
                response_parts.append(
                    f"{i}. {title} (Quality: {quality:.2f})"
                )
                response_parts.append(
                    f"   This reference provides insights into the research question "
                    f"with a confidence level of {quality:.2f}."
                )
                response_parts.append("")
        
        response_parts.extend([
            "Conclusion:",
            f"The analysis of {len(relevant_refs)} high-quality references provides "
            f"comprehensive insights into {query}. The research demonstrates strong "
            f"evidence-based findings with robust provenance tracking.",
            "",
            f"Research conducted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(response_parts)
    
    async def _calculate_research_quality(self, query: str, relevant_refs: List[str], 
                                        response: str, query_type: str) -> Dict[str, float]:
        """Calculate comprehensive research quality metrics"""
        
        # Use the enhanced quality calculation
        base_quality = calculate_research_quality_score(
            self.reference_pipeline, relevant_refs, response
        )
        
        # Additional quality factors
        reference_diversity = len(set(
            self._get_source_for_reference(ref_id) 
            for ref_id in relevant_refs
        )) / max(1, len(relevant_refs))
        
        # Calculate average reference quality
        ref_qualities = []
        for ref_id in relevant_refs:
            provenance = self.reference_pipeline.get_reference_provenance(ref_id)
            if provenance and provenance.get("provenance_trace"):
                quality = provenance["provenance_trace"]["quality_metrics"].get("composite_quality", 0.5)
                ref_qualities.append(quality)
        
        avg_ref_quality = sum(ref_qualities) / len(ref_qualities) if ref_qualities else 0.5
        
        # Response completeness (based on length and structure)
        response_completeness = min(1.0, len(response.split()) / 200.0)
        
        # Calculate overall quality
        overall_quality = (
            base_quality * 0.4 +
            reference_diversity * 0.2 +
            avg_ref_quality * 0.3 +
            response_completeness * 0.1
        )
        
        return {
            "overall_quality": overall_quality,
            "base_quality": base_quality,
            "reference_diversity": reference_diversity,
            "average_reference_quality": avg_ref_quality,
            "response_completeness": response_completeness,
            "num_references": len(relevant_refs)
        }
    
    def _generate_rlhf_rewards(self, query_type: str) -> Dict[str, float]:
        """Generate RLHF reward signals"""
        
        research_context = {
            "query_type": query_type,
            "session_id": self.session_id,
            "research_count": len(self.research_history)
        }
        
        return get_rlhf_reward_signals(self.reference_pipeline, research_context)
    
    def _get_provenance_for_references(self, ref_ids: List[str]) -> List[Dict[str, Any]]:
        """Get provenance information for a list of references"""
        
        provenance_data = []
        
        for ref_id in ref_ids:
            provenance = self.reference_pipeline.get_reference_provenance(ref_id)
            if provenance:
                provenance_data.append({
                    "reference_id": ref_id,
                    "trace_id": provenance.get("provenance_trace", {}).get("trace_id"),
                    "quality_score": provenance.get("quality_metrics", {}).get("composite_quality", 0.5),
                    "reward_signals": provenance.get("reward_signals", {}),
                    "reasoning_steps": provenance.get("provenance_trace", {}).get("reasoning_steps", [])
                })
        
        return provenance_data
    
    def _get_source_for_reference(self, ref_id: str) -> str:
        """Get source type for a reference"""
        
        provenance = self.reference_pipeline.get_reference_provenance(ref_id)
        if provenance and provenance.get("provenance_trace"):
            for step in provenance["provenance_trace"]["reasoning_steps"]:
                if "ingested from" in step and "source" in step:
                    try:
                        return step.split("ingested from")[1].split("source")[0].strip()
                    except:
                        pass
        
        return "unknown"
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get comprehensive session analytics"""
        
        if not self.research_history:
            return {"message": "No research conducted yet"}
        
        # Calculate session metrics
        total_queries = len(self.research_history)
        avg_quality = sum(self.quality_metrics) / len(self.quality_metrics)
        
        # Quality trend
        quality_trend = "stable"
        if len(self.quality_metrics) >= 2:
            recent_avg = sum(self.quality_metrics[-2:]) / 2
            early_avg = sum(self.quality_metrics[:2]) / min(2, len(self.quality_metrics))
            
            if recent_avg > early_avg + 0.1:
                quality_trend = "improving"
            elif recent_avg < early_avg - 0.1:
                quality_trend = "declining"
        
        # Reference usage statistics
        all_refs_used = []
        for record in self.research_history:
            all_refs_used.extend(record["relevant_references"])
        
        unique_refs = len(set(all_refs_used))
        total_refs_used = len(all_refs_used)
        
        # RLHF reward analysis
        all_rlhf_rewards = [record["rlhf_rewards"] for record in self.research_history]
        avg_rlhf_rewards = {}
        
        if all_rlhf_rewards:
            for reward_type in all_rlhf_rewards[0].keys():
                avg_rlhf_rewards[reward_type] = sum(
                    rewards[reward_type] for rewards in all_rlhf_rewards
                ) / len(all_rlhf_rewards)
        
        return {
            "session_id": self.session_id,
            "total_queries": total_queries,
            "average_quality": avg_quality,
            "quality_trend": quality_trend,
            "unique_references_used": unique_refs,
            "total_references_used": total_refs_used,
            "reference_reuse_rate": (total_refs_used - unique_refs) / max(1, total_refs_used),
            "average_rlhf_rewards": avg_rlhf_rewards,
            "research_history": self.research_history[-3:]  # Last 3 queries for brevity
        }
    
    def export_session_data(self, output_file: str = None) -> str:
        """Export session data for analysis"""
        
        if output_file is None:
            output_file = f"research_session_{self.session_id}.json"
        
        session_data = {
            "session_analytics": self.get_session_analytics(),
            "reference_pipeline_stats": self.reference_pipeline.get_ingestion_statistics(),
            "trace_quality_dashboard": self.reference_pipeline.get_trace_quality_dashboard(),
            "full_research_history": self.research_history
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"📄 Session data exported to {output_file}")
        return output_file

async def main():
    """Demonstration of enhanced reference integration"""
    
    print("🚀 Enhanced Reference Integration Demo")
    print("=" * 60)
    
    # Initialize enhanced research agent
    agent = EnhancedResearchAgent()
    
    # Load knowledge base
    success = await agent.load_knowledge_base()
    if not success:
        print("❌ Failed to load knowledge base")
        return
    
    # Conduct multiple research queries
    research_queries = [
        ("How do transformer architectures work?", "foundational_concepts"),
        ("What are recent advances in reinforcement learning?", "recent_research"),
        ("How can I implement a neural network from scratch?", "practical_implementation"),
        ("What is the state of the art in natural language processing?", "comprehensive_survey")
    ]
    
    print(f"\n🔬 Conducting {len(research_queries)} research queries...")
    
    for query, query_type in research_queries:
        result = await agent.conduct_research(query, query_type)
        
        # Show brief results
        print(f"\n📊 Query: '{query[:50]}...'")
        print(f"   Quality: {result['quality_metrics']['overall_quality']:.3f}")
        print(f"   References: {len(result['relevant_references'])}")
        print(f"   RLHF Reward: {result['rlhf_rewards']['reference_quality_reward']:.3f}")
    
    # Show session analytics
    print(f"\n📈 Session Analytics")
    print("=" * 40)
    
    analytics = agent.get_session_analytics()
    
    print(f"Total Queries: {analytics['total_queries']}")
    print(f"Average Quality: {analytics['average_quality']:.3f}")
    print(f"Quality Trend: {analytics['quality_trend']}")
    print(f"Unique References Used: {analytics['unique_references_used']}")
    print(f"Reference Reuse Rate: {analytics['reference_reuse_rate']:.1%}")
    
    print(f"\nTop RLHF Rewards:")
    for reward_type, value in analytics['average_rlhf_rewards'].items():
        print(f"   {reward_type}: {value:.3f}")
    
    # Export session data
    export_file = agent.export_session_data()
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📄 Full session data available in: {export_file}")
    
    # Show integration benefits
    print(f"\n✨ Enhanced Integration Benefits Demonstrated:")
    print(f"   ✅ Trace-level provenance for all research")
    print(f"   ✅ Quality-based reference selection")
    print(f"   ✅ RLHF reward signal generation")
    print(f"   ✅ Comprehensive quality metrics")
    print(f"   ✅ Session-level analytics and trends")
    print(f"   ✅ Semantic graph enhancement")
    print(f"   ✅ Citation network analysis")

if __name__ == "__main__":
    asyncio.run(main())
