# -*- coding: utf-8 -*-
"""
Reference Ingestion Pipeline
Ingests structured references and maps them into semantic graph with citation metadata,
enabling trace-level provenance and reward shaping.
"""

import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import networkx as nx
import numpy as np
from collections import defaultdict

# Import semantic graph components
try:
    from .stage_3_semantic_graph import (
        SemanticGraphManager, NodeType, EdgeType, SourceType, 
        GraphNode, GraphEdge
    )
    SEMANTIC_GRAPH_AVAILABLE = True
except ImportError:
    try:
        from stage_3_semantic_graph import (
            SemanticGraphManager, NodeType, EdgeType, SourceType, 
            GraphNode, GraphEdge
        )
        SEMANTIC_GRAPH_AVAILABLE = True
    except ImportError:
        SEMANTIC_GRAPH_AVAILABLE = False
        # Create dummy classes for type hints
        class SemanticGraphManager: pass
        class NodeType: pass
        class EdgeType: pass
        class SourceType: pass
        class GraphNode: pass
        class GraphEdge: pass
        print("⚠️ Semantic graph not available, using basic NetworkX implementation")

@dataclass
class CitationMetadata:
    """Rich citation metadata for provenance tracking"""
    citation_id: str
    source_reference: str
    target_reference: str
    citation_type: str  # "direct", "indirect", "supporting", "contradicting"
    confidence_score: float
    context_snippet: str
    page_number: Optional[int]
    section: Optional[str]
    extracted_at: datetime
    validation_status: str  # "verified", "pending", "disputed"

@dataclass
class ProvenanceTrace:
    """Trace-level provenance information"""
    trace_id: str
    reference_chain: List[str]  # Chain of reference IDs
    confidence_path: List[float]  # Confidence at each step
    reasoning_steps: List[str]
    reward_signals: Dict[str, float]
    quality_metrics: Dict[str, float]
    created_at: datetime

class ReferenceIngestionPipeline:
    """Enhanced reference ingestion with semantic graph integration"""
    
    def __init__(self, semantic_graph: Optional[SemanticGraphManager] = None,
                 storage_path: str = "reference_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize semantic graph
        if semantic_graph:
            self.semantic_graph = semantic_graph
        elif SEMANTIC_GRAPH_AVAILABLE:
            self.semantic_graph = SemanticGraphManager()
        else:
            self.semantic_graph = None
            self.fallback_graph = nx.MultiDiGraph()
        
        # Citation and provenance tracking
        self.citations: Dict[str, CitationMetadata] = {}
        self.provenance_traces: Dict[str, ProvenanceTrace] = {}
        self.reference_mappings: Dict[str, str] = {}  # external_id -> internal_node_id
        
        # Quality metrics
        self.ingestion_stats = {
            "references_processed": 0,
            "citations_extracted": 0,
            "provenance_traces_created": 0,
            "duplicate_references_merged": 0,
            "validation_errors": 0
        }
        
        print("🔗 Reference Ingestion Pipeline initialized")
        if self.semantic_graph:
            print("   ✅ Semantic graph integration enabled")
        else:
            print("   ⚠️ Using fallback NetworkX graph")
    
    def load_references(self, path: str) -> List[Dict]:
        """Load references from JSON file with validation"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                references = json.load(f)
            
            # Validate reference structure
            validated_refs = []
            for ref in references:
                if self._validate_reference_structure(ref):
                    validated_refs.append(ref)
                else:
                    self.ingestion_stats["validation_errors"] += 1
                    print(f"⚠️ Invalid reference structure: {ref.get('id', 'unknown')}")
            
            print(f"📚 Loaded {len(validated_refs)} valid references from {path}")
            return validated_refs
            
        except Exception as e:
            print(f"❌ Failed to load references from {path}: {e}")
            return []
    
    def ingest_references(self, references: List[Dict], 
                         extract_citations: bool = True,
                         create_provenance: bool = True) -> Dict[str, Any]:
        """
        Ingest references into semantic graph with full metadata
        
        Args:
            references: List of reference dictionaries
            extract_citations: Whether to extract citation relationships
            create_provenance: Whether to create provenance traces
            
        Returns:
            Ingestion statistics and results
        """
        ingestion_results = {
            "nodes_created": [],
            "citations_extracted": [],
            "provenance_traces": [],
            "errors": []
        }
        
        print(f"🔄 Ingesting {len(references)} references...")
        
        # Phase 1: Create reference nodes
        for ref in references:
            try:
                node_id = self._create_reference_node(ref)
                if node_id:
                    ingestion_results["nodes_created"].append(node_id)
                    self.ingestion_stats["references_processed"] += 1
            except Exception as e:
                error_msg = f"Failed to create node for {ref.get('id', 'unknown')}: {e}"
                ingestion_results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Phase 2: Extract citations if enabled
        if extract_citations:
            for ref in references:
                try:
                    citations = self._extract_citations_from_reference(ref)
                    ingestion_results["citations_extracted"].extend(citations)
                except Exception as e:
                    error_msg = f"Failed to extract citations for {ref.get('id', 'unknown')}: {e}"
                    ingestion_results["errors"].append(error_msg)
        
        # Phase 3: Create provenance traces if enabled
        if create_provenance:
            for ref in references:
                try:
                    trace = self._create_provenance_trace(ref)
                    if trace:
                        ingestion_results["provenance_traces"].append(trace.trace_id)
                except Exception as e:
                    error_msg = f"Failed to create provenance for {ref.get('id', 'unknown')}: {e}"
                    ingestion_results["errors"].append(error_msg)
        
        # Phase 4: Update reward signals
        self._update_reward_signals(ingestion_results)
        
        print(f"✅ Reference ingestion complete:")
        print(f"   Nodes created: {len(ingestion_results['nodes_created'])}")
        print(f"   Citations extracted: {len(ingestion_results['citations_extracted'])}")
        print(f"   Provenance traces: {len(ingestion_results['provenance_traces'])}")
        print(f"   Errors: {len(ingestion_results['errors'])}")
        
        return ingestion_results
    
    def _create_reference_node(self, ref: Dict) -> Optional[str]:
        """Create a reference node in the semantic graph"""
        
        # Extract node data
        node_data = self._map_reference_to_node_data(ref)
        
        if self.semantic_graph:
            # Use semantic graph
            node_id = self.semantic_graph.add_node(
                content=node_data["content"],
                node_type=node_data["node_type"],
                source_type=node_data["source_type"],
                title=node_data.get("title"),
                source_id=node_data.get("source_id"),
                metadata=node_data.get("metadata", {}),
                tags=node_data.get("tags", []),
                importance_score=node_data.get("importance_score", 0.5),
                confidence_score=node_data.get("confidence_score", 0.8)
            )
        else:
            # Use fallback NetworkX graph
            node_id = f"ref_{ref['id']}"
            self.fallback_graph.add_node(node_id, **node_data)
        
        # Store mapping
        self.reference_mappings[ref["id"]] = node_id
        
        return node_id
    
    def _map_reference_to_node_data(self, ref: Dict) -> Dict[str, Any]:
        """Map reference dictionary to semantic graph node data"""
        
        # Determine node type based on reference type
        ref_type = ref.get("type", "PAPER").upper()
        node_type_mapping = {
            "PAPER": NodeType.PAPER if SEMANTIC_GRAPH_AVAILABLE else "PAPER",
            "CODE": NodeType.CODE_SNIPPET if SEMANTIC_GRAPH_AVAILABLE else "CODE",
            "DATASET": NodeType.DATASET if SEMANTIC_GRAPH_AVAILABLE else "DATASET",
            "CONCEPT": NodeType.CONCEPT if SEMANTIC_GRAPH_AVAILABLE else "CONCEPT"
        }
        node_type = node_type_mapping.get(ref_type, NodeType.PAPER if SEMANTIC_GRAPH_AVAILABLE else "PAPER")
        
        # Determine source type
        source = ref.get("source", "ARXIV").upper()
        source_type_mapping = {
            "ARXIV": SourceType.ARXIV if SEMANTIC_GRAPH_AVAILABLE else "ARXIV",
            "GITHUB": SourceType.GITHUB if SEMANTIC_GRAPH_AVAILABLE else "GITHUB",
            "PUBMED": SourceType.PUBMED if SEMANTIC_GRAPH_AVAILABLE else "PUBMED",
            "WIKIPEDIA": SourceType.WIKIPEDIA if SEMANTIC_GRAPH_AVAILABLE else "WIKIPEDIA"
        }
        source_type = source_type_mapping.get(source, SourceType.INTERNAL if SEMANTIC_GRAPH_AVAILABLE else "INTERNAL")
        
        # Create content from available fields
        content_parts = []
        if ref.get("title"):
            content_parts.append(f"Title: {ref['title']}")
        if ref.get("abstract"):
            content_parts.append(f"Abstract: {ref['abstract']}")
        elif ref.get("description"):
            content_parts.append(f"Description: {ref['description']}")
        
        content = "\n".join(content_parts) if content_parts else str(ref)
        
        # Calculate importance score based on citations and other factors
        citation_count = ref.get("citations", 0)
        importance_score = min(0.9, 0.3 + (citation_count / 100.0))  # Scale citations to importance
        
        # Calculate confidence score based on source and metadata completeness
        confidence_score = ref.get("confidence", 0.8)
        if ref.get("authors") and ref.get("published"):
            confidence_score = min(0.95, confidence_score + 0.1)
        
        return {
            "content": content,
            "title": ref.get("title"),
            "node_type": node_type,
            "source_type": source_type,
            "source_id": ref.get("id"),
            "metadata": {
                "authors": ref.get("authors", []),
                "published": ref.get("published", "unknown"),
                "citation_count": citation_count,
                "doi": ref.get("doi"),
                "url": ref.get("url"),
                "venue": ref.get("venue"),
                "keywords": ref.get("keywords", []),
                "abstract": ref.get("abstract"),
                "ingestion_timestamp": datetime.now().isoformat()
            },
            "tags": ref.get("tags", []) + [ref.get("source", "").lower()],
            "importance_score": importance_score,
            "confidence_score": confidence_score
        }
    
    def _extract_citations_from_reference(self, ref: Dict) -> List[str]:
        """Extract citation relationships from reference"""
        
        citations_created = []
        ref_node_id = self.reference_mappings.get(ref["id"])
        
        if not ref_node_id:
            return citations_created
        
        # Extract direct citations from references field
        references = ref.get("references", [])
        for cited_ref in references:
            citation_id = self._create_citation_relationship(
                source_ref=ref["id"],
                target_ref=cited_ref,
                citation_type="direct",
                confidence=0.9
            )
            if citation_id:
                citations_created.append(citation_id)
        
        # Extract author relationships
        authors = ref.get("authors", [])
        for author in authors:
            author_node_id = self._find_or_create_author_node(author)
            if author_node_id and self.semantic_graph:
                edge_id = self.semantic_graph.add_edge(
                    source_node=author_node_id,
                    target_node=ref_node_id,
                    edge_type=EdgeType.AUTHORED_BY,
                    confidence=0.95,
                    metadata={"relationship_type": "authorship"}
                )
                citations_created.append(edge_id)
        
        # Extract venue/institution relationships
        venue = ref.get("venue")
        if venue:
            venue_node_id = self._find_or_create_venue_node(venue)
            if venue_node_id and self.semantic_graph:
                edge_id = self.semantic_graph.add_edge(
                    source_node=ref_node_id,
                    target_node=venue_node_id,
                    edge_type=EdgeType.PART_OF,
                    confidence=0.8,
                    metadata={"relationship_type": "publication_venue"}
                )
                citations_created.append(edge_id)
        
        self.ingestion_stats["citations_extracted"] += len(citations_created)
        return citations_created
    
    def _create_citation_relationship(self, source_ref: str, target_ref: str,
                                    citation_type: str, confidence: float) -> Optional[str]:
        """Create a citation relationship between two references"""
        
        source_node_id = self.reference_mappings.get(source_ref)
        target_node_id = self.reference_mappings.get(target_ref)
        
        if not source_node_id or not target_node_id:
            return None
        
        # Create citation metadata
        citation_id = str(uuid.uuid4())
        citation_metadata = CitationMetadata(
            citation_id=citation_id,
            source_reference=source_ref,
            target_reference=target_ref,
            citation_type=citation_type,
            confidence_score=confidence,
            context_snippet="",  # Could be extracted from full text
            page_number=None,
            section=None,
            extracted_at=datetime.now(),
            validation_status="pending"
        )
        
        self.citations[citation_id] = citation_metadata
        
        # Create edge in graph
        if self.semantic_graph:
            edge_id = self.semantic_graph.add_edge(
                source_node=source_node_id,
                target_node=target_node_id,
                edge_type=EdgeType.CITES,
                confidence=confidence,
                metadata={
                    "citation_id": citation_id,
                    "citation_type": citation_type,
                    "extracted_at": datetime.now().isoformat()
                }
            )
        else:
            edge_id = citation_id
            self.fallback_graph.add_edge(
                source_node_id, target_node_id,
                citation_id=citation_id,
                citation_type=citation_type,
                confidence=confidence
            )
        
        return edge_id
    
    def _create_provenance_trace(self, ref: Dict) -> Optional[ProvenanceTrace]:
        """Create comprehensive provenance trace for reference"""
        
        trace_id = str(uuid.uuid4())
        ref_node_id = self.reference_mappings.get(ref["id"])
        
        if not ref_node_id:
            return None
        
        # Build enhanced reference chain with citation analysis
        reference_chain = [ref["id"]]
        confidence_path = [ref.get("confidence", 0.8)]
        
        # Add cited references to chain if they exist in our dataset
        cited_refs = ref.get("references", [])
        for cited_ref in cited_refs:
            if cited_ref in self.reference_mappings:
                reference_chain.append(cited_ref)
                # Estimate confidence based on citation relationship
                confidence_path.append(0.7)  # Lower confidence for indirect references
        
        # Create detailed reasoning steps
        reasoning_steps = [
            f"Reference {ref['id']} ingested from {ref.get('source', 'unknown')} source",
            f"Initial confidence score: {ref.get('confidence', 0.8)}",
            f"Citation count: {ref.get('citations', 0)}",
            f"Publication year: {ref.get('published', 'unknown')}",
            f"Authors: {len(ref.get('authors', []))} author(s)",
            f"Reference type: {ref.get('type', 'unknown')}"
        ]
        
        # Add citation network analysis
        if cited_refs:
            reasoning_steps.append(f"Cites {len(cited_refs)} references")
            internal_citations = sum(1 for cr in cited_refs if cr in self.reference_mappings)
            if internal_citations > 0:
                reasoning_steps.append(f"Has {internal_citations} internal citations (cross-references)")
        
        # Add venue/source analysis
        venue = ref.get("venue")
        if venue:
            reasoning_steps.append(f"Published in: {venue}")
        
        # Calculate reward signals based on reference quality
        reward_signals = self._calculate_reference_reward_signals(ref)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_reference_quality_metrics(ref)
        
        # Add composite quality score
        composite_quality = self._calculate_composite_quality_score(reward_signals, quality_metrics)
        quality_metrics["composite_quality"] = composite_quality
        
        # Add trace-level metadata
        trace_metadata = {
            "ingestion_timestamp": datetime.now().isoformat(),
            "reference_type": ref.get("type", "unknown"),
            "source_type": ref.get("source", "unknown"),
            "has_semantic_graph": self.semantic_graph is not None,
            "citation_network_size": len(cited_refs),
            "internal_citations": sum(1 for cr in cited_refs if cr in self.reference_mappings)
        }
        
        reasoning_steps.append(f"Composite quality score: {composite_quality:.3f}")
        
        provenance_trace = ProvenanceTrace(
            trace_id=trace_id,
            reference_chain=reference_chain,
            confidence_path=confidence_path,
            reasoning_steps=reasoning_steps,
            reward_signals=reward_signals,
            quality_metrics=quality_metrics,
            created_at=datetime.now()
        )
        
        # Store trace with metadata (add as attribute)
        setattr(provenance_trace, 'metadata', trace_metadata)
        self.provenance_traces[trace_id] = provenance_trace
        self.ingestion_stats["provenance_traces_created"] += 1
        
        return provenance_trace
    
    def _calculate_reference_reward_signals(self, ref: Dict) -> Dict[str, float]:
        """Calculate comprehensive reward signals for reference quality"""
        
        signals = {}
        
        # Citation-based reward (enhanced with logarithmic scaling)
        citation_count = ref.get("citations", 0)
        if citation_count > 0:
            # Use log scaling for better distribution
            signals["citation_reward"] = min(1.0, np.log10(citation_count + 1) / 5.0)
        else:
            signals["citation_reward"] = 0.1  # Small baseline for uncited works
        
        # Completeness reward (enhanced with weighted fields)
        field_weights = {
            "title": 0.2, "authors": 0.2, "published": 0.15,
            "abstract": 0.15, "doi": 0.1, "venue": 0.1, "keywords": 0.1
        }
        completeness = sum(
            weight for field, weight in field_weights.items() 
            if ref.get(field)
        )
        signals["completeness_reward"] = completeness
        
        # Recency reward (enhanced with different decay rates by field)
        try:
            published_year = int(ref.get("published", "2000")[:4])
            current_year = datetime.now().year
            age = current_year - published_year
            
            # Different decay rates for different fields
            ref_type = ref.get("type", "PAPER").upper()
            if ref_type == "CODE":
                # Code becomes obsolete faster
                recency = max(0, 1 - age / 5.0)
            elif ref_type == "DATASET":
                # Datasets have longer relevance
                recency = max(0, 1 - age / 15.0)
            else:
                # Papers have medium decay
                recency = max(0, 1 - age / 10.0)
            
            signals["recency_reward"] = recency
        except:
            signals["recency_reward"] = 0.5
        
        # Source credibility reward (enhanced with venue consideration)
        base_credibility = {
            "ARXIV": 0.8, "PUBMED": 0.9, "GITHUB": 0.7, 
            "WIKIPEDIA": 0.6, "SEMANTIC_SCHOLAR": 0.85
        }
        source_cred = base_credibility.get(ref.get("source", "").upper(), 0.5)
        
        # Boost credibility for prestigious venues
        venue = ref.get("venue", "").lower()
        prestigious_venues = {
            "nature", "science", "cell", "neurips", "icml", "iclr", 
            "acl", "emnlp", "cvpr", "iccv", "eccv"
        }
        if any(pv in venue for pv in prestigious_venues):
            source_cred = min(1.0, source_cred + 0.1)
        
        signals["source_credibility"] = source_cred
        
        # Network centrality reward (based on reference connections)
        references = ref.get("references", [])
        signals["network_centrality"] = min(1.0, len(references) / 20.0)
        
        # Author authority reward (based on author count and h-index proxy)
        authors = ref.get("authors", [])
        author_count = len(authors)
        # More authors can indicate collaboration but diminishing returns
        signals["author_authority"] = min(1.0, np.log10(author_count + 1) / 2.0)
        
        # Content richness reward (based on abstract/description quality)
        content = ref.get("abstract", ref.get("description", ""))
        if content:
            # Reward longer, more detailed content
            content_length = len(content.split())
            signals["content_richness"] = min(1.0, content_length / 200.0)
        else:
            signals["content_richness"] = 0.0
        
        # Cross-reference reward (bonus for citing other references in our dataset)
        cross_refs = 0
        for cited_ref in references:
            if cited_ref in self.reference_mappings:
                cross_refs += 1
        signals["cross_reference_reward"] = min(1.0, cross_refs / 5.0)
        
        return signals
    
    def _calculate_reference_quality_metrics(self, ref: Dict) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for reference"""
        
        metrics = {}
        
        # Enhanced metadata completeness with field importance weighting
        field_importance = {
            "title": 0.2, "authors": 0.15, "published": 0.15, "abstract": 0.15,
            "doi": 0.1, "venue": 0.1, "keywords": 0.05, "url": 0.05, "type": 0.05
        }
        
        weighted_completeness = sum(
            weight for field, weight in field_importance.items() 
            if ref.get(field)
        )
        metrics["metadata_completeness"] = weighted_completeness
        
        # Content richness (enhanced with multiple content sources)
        content_sources = [
            ref.get("abstract", ""),
            ref.get("description", ""),
            " ".join(ref.get("keywords", [])),
            ref.get("title", "")
        ]
        total_content = " ".join(filter(None, content_sources))
        content_words = len(total_content.split())
        metrics["content_richness"] = min(1.0, content_words / 300.0)  # Normalize to 300 words
        
        # Authority score (enhanced with multiple factors)
        citation_count = ref.get("citations", 0)
        author_count = len(ref.get("authors", []))
        
        # Citation component (logarithmic scaling)
        citation_component = min(0.7, np.log10(citation_count + 1) / 5.0) if citation_count > 0 else 0.0
        
        # Author component (collaboration indicator)
        author_component = min(0.2, author_count / 10.0)
        
        # Venue component (prestigious venue bonus)
        venue = ref.get("venue", "").lower()
        venue_component = 0.1 if any(pv in venue for pv in [
            "nature", "science", "cell", "neurips", "icml", "iclr", "acl", "emnlp"
        ]) else 0.0
        
        metrics["authority_score"] = citation_component + author_component + venue_component
        
        # Network connectivity (based on references and potential citations)
        references = ref.get("references", [])
        metrics["network_connectivity"] = min(1.0, len(references) / 15.0)
        
        # Temporal relevance (age-adjusted quality)
        try:
            published_year = int(ref.get("published", "2000")[:4])
            current_year = datetime.now().year
            age = current_year - published_year
            
            # Different relevance decay for different types
            ref_type = ref.get("type", "PAPER").upper()
            if ref_type == "CODE":
                temporal_relevance = max(0.1, 1 - age / 3.0)  # Code ages fast
            elif ref_type == "DATASET":
                temporal_relevance = max(0.3, 1 - age / 10.0)  # Datasets age slower
            else:
                temporal_relevance = max(0.2, 1 - age / 7.0)  # Papers medium aging
            
            metrics["temporal_relevance"] = temporal_relevance
        except:
            metrics["temporal_relevance"] = 0.5
        
        # Source reliability (based on source type and venue)
        source_reliability = {
            "ARXIV": 0.8, "PUBMED": 0.9, "GITHUB": 0.7, 
            "WIKIPEDIA": 0.6, "SEMANTIC_SCHOLAR": 0.85
        }
        base_reliability = source_reliability.get(ref.get("source", "").upper(), 0.5)
        
        # Adjust for DOI presence (indicates peer review)
        if ref.get("doi"):
            base_reliability = min(1.0, base_reliability + 0.1)
        
        metrics["source_reliability"] = base_reliability
        
        # Cross-validation score (how well it connects to other references)
        cited_refs = ref.get("references", [])
        internal_citations = sum(1 for cr in cited_refs if cr in self.reference_mappings)
        metrics["cross_validation_score"] = min(1.0, internal_citations / 3.0)
        
        return metrics
    
    def _calculate_composite_quality_score(self, reward_signals: Dict[str, float], 
                                         quality_metrics: Dict[str, float]) -> float:
        """Calculate composite quality score from reward signals and quality metrics"""
        
        # Weight different components
        reward_weights = {
            "citation_reward": 0.25,
            "completeness_reward": 0.15,
            "recency_reward": 0.10,
            "source_credibility": 0.15,
            "network_centrality": 0.10,
            "author_authority": 0.10,
            "content_richness": 0.10,
            "cross_reference_reward": 0.05
        }
        
        metric_weights = {
            "metadata_completeness": 0.20,
            "authority_score": 0.25,
            "network_connectivity": 0.15,
            "temporal_relevance": 0.15,
            "source_reliability": 0.15,
            "cross_validation_score": 0.10
        }
        
        # Calculate weighted reward score
        reward_score = sum(
            reward_signals.get(signal, 0.0) * weight
            for signal, weight in reward_weights.items()
        )
        
        # Calculate weighted metric score
        metric_score = sum(
            quality_metrics.get(metric, 0.0) * weight
            for metric, weight in metric_weights.items()
        )
        
        # Combine with slight preference for metrics (more objective)
        composite_score = 0.4 * reward_score + 0.6 * metric_score
        
        return min(1.0, max(0.0, composite_score))
    
    def _update_reward_signals(self, ingestion_results: Dict[str, Any]):
        """Update reward signals based on ingestion results"""
        
        # Calculate overall ingestion quality
        total_nodes = len(ingestion_results["nodes_created"])
        total_citations = len(ingestion_results["citations_extracted"])
        total_errors = len(ingestion_results["errors"])
        
        if total_nodes > 0:
            error_rate = total_errors / total_nodes
            citation_rate = total_citations / total_nodes
            
            # Update global reward signals
            ingestion_quality = max(0, 1.0 - error_rate) * min(1.0, citation_rate)
            
            print(f"📊 Ingestion Quality Score: {ingestion_quality:.3f}")
            print(f"   Error Rate: {error_rate:.3f}")
            print(f"   Citation Rate: {citation_rate:.3f}")
    
    def get_reference_provenance(self, reference_id: str) -> Optional[Dict[str, Any]]:
        """Get complete provenance information for a reference"""
        
        node_id = self.reference_mappings.get(reference_id)
        if not node_id:
            return None
        
        # Find associated provenance trace
        trace = None
        for t in self.provenance_traces.values():
            if reference_id in t.reference_chain:
                trace = t
                break
        
        # Get citations involving this reference
        related_citations = []
        for citation in self.citations.values():
            if (citation.source_reference == reference_id or 
                citation.target_reference == reference_id):
                related_citations.append(asdict(citation))
        
        provenance_info = {
            "reference_id": reference_id,
            "node_id": node_id,
            "provenance_trace": asdict(trace) if trace else None,
            "related_citations": related_citations,
            "quality_metrics": trace.quality_metrics if trace else {},
            "reward_signals": trace.reward_signals if trace else {}
        }
        
        return provenance_info
    
    def get_citation_network(self, reference_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get citation network around a reference"""
        
        if not self.semantic_graph:
            return {"error": "Semantic graph not available"}
        
        node_id = self.reference_mappings.get(reference_id)
        if not node_id:
            return {"error": "Reference not found"}
        
        # Get neighbors in citation network
        neighbors = self.semantic_graph.get_node_neighbors(
            node_id, 
            max_depth=max_depth,
            edge_types=[EdgeType.CITES, EdgeType.AUTHORED_BY] if SEMANTIC_GRAPH_AVAILABLE else None
        )
        
        # Build citation network
        network = {
            "center_reference": reference_id,
            "neighbors": {},
            "citation_paths": []
        }
        
        for neighbor_id, neighbor_info in neighbors.items():
            # Find original reference ID
            original_ref_id = None
            for ref_id, mapped_id in self.reference_mappings.items():
                if mapped_id == neighbor_id:
                    original_ref_id = ref_id
                    break
            
            if original_ref_id:
                network["neighbors"][original_ref_id] = {
                    "distance": neighbor_info["distance"],
                    "relationship": neighbor_info["relationship"],
                    "confidence": neighbor_info["confidence"]
                }
        
        return network
    
    def _validate_reference_structure(self, ref: Dict) -> bool:
        """Validate reference structure"""
        required_fields = ["id"]
        return all(field in ref for field in required_fields)
    
    def _find_or_create_author_node(self, author: str) -> Optional[str]:
        """Find or create author node"""
        if not self.semantic_graph:
            return None
        
        # Simple author node creation (could be enhanced with author disambiguation)
        author_id = self.semantic_graph.add_node(
            content=f"Author: {author}",
            node_type=NodeType.AUTHOR,
            source_type=SourceType.INTERNAL,
            title=author,
            metadata={"author_name": author}
        )
        
        return author_id
    
    def _find_or_create_venue_node(self, venue: str) -> Optional[str]:
        """Find or create venue node"""
        if not self.semantic_graph:
            return None
        
        venue_id = self.semantic_graph.add_node(
            content=f"Venue: {venue}",
            node_type=NodeType.INSTITUTION,
            source_type=SourceType.INTERNAL,
            title=venue,
            metadata={"venue_name": venue}
        )
        
        return venue_id
    
    def enhance_semantic_graph_with_provenance(self) -> Dict[str, Any]:
        """Enhance semantic graph nodes with provenance and reward information"""
        
        if not self.semantic_graph:
            return {"error": "Semantic graph not available"}
        
        enhancement_stats = {
            "nodes_enhanced": 0,
            "provenance_links_added": 0,
            "reward_annotations_added": 0
        }
        
        # Enhance each reference node with provenance data
        for ref_id, node_id in self.reference_mappings.items():
            if node_id in self.semantic_graph.nodes:
                node = self.semantic_graph.nodes[node_id]
                
                # Find associated provenance trace
                trace = None
                for t in self.provenance_traces.values():
                    if ref_id in t.reference_chain:
                        trace = t
                        break
                
                if trace:
                    # Add provenance metadata to node
                    node.metadata.update({
                        "provenance_trace_id": trace.trace_id,
                        "composite_quality": trace.quality_metrics.get("composite_quality", 0.5),
                        "reward_signals": trace.reward_signals,
                        "quality_metrics": trace.quality_metrics,
                        "reasoning_steps": trace.reasoning_steps,
                        "trace_created_at": trace.created_at.isoformat()
                    })
                    
                    # Update importance score based on composite quality
                    composite_quality = trace.quality_metrics.get("composite_quality", 0.5)
                    node.importance_score = max(node.importance_score, composite_quality)
                    
                    # Add quality-based tags
                    quality_tags = []
                    if composite_quality > 0.8:
                        quality_tags.append("high_quality")
                    elif composite_quality > 0.6:
                        quality_tags.append("medium_quality")
                    else:
                        quality_tags.append("low_quality")
                    
                    # Add reward-based tags
                    if trace.reward_signals.get("citation_reward", 0) > 0.7:
                        quality_tags.append("highly_cited")
                    if trace.reward_signals.get("recency_reward", 0) > 0.8:
                        quality_tags.append("recent")
                    if trace.reward_signals.get("source_credibility", 0) > 0.8:
                        quality_tags.append("credible_source")
                    
                    node.tags.extend(quality_tags)
                    node.tags = list(set(node.tags))  # Remove duplicates
                    
                    enhancement_stats["nodes_enhanced"] += 1
                    enhancement_stats["reward_annotations_added"] += 1
        
        # Create provenance links between related references
        for trace in self.provenance_traces.values():
            if len(trace.reference_chain) > 1:
                for i in range(len(trace.reference_chain) - 1):
                    source_ref = trace.reference_chain[i]
                    target_ref = trace.reference_chain[i + 1]
                    
                    source_node = self.reference_mappings.get(source_ref)
                    target_node = self.reference_mappings.get(target_ref)
                    
                    if source_node and target_node:
                        # Add provenance edge
                        edge_id = self.semantic_graph.add_edge(
                            source_node=source_node,
                            target_node=target_node,
                            edge_type=EdgeType.DERIVED_FROM if SEMANTIC_GRAPH_AVAILABLE else "DERIVED_FROM",
                            confidence=trace.confidence_path[i] if i < len(trace.confidence_path) else 0.7,
                            metadata={
                                "provenance_trace_id": trace.trace_id,
                                "relationship_type": "provenance_chain",
                                "chain_position": i
                            }
                        )
                        enhancement_stats["provenance_links_added"] += 1
        
        return enhancement_stats
    
    def get_trace_quality_dashboard(self) -> Dict[str, Any]:
        """Generate a comprehensive trace quality dashboard"""
        
        dashboard = {
            "overview": {
                "total_traces": len(self.provenance_traces),
                "total_references": len(self.reference_mappings),
                "total_citations": len(self.citations)
            },
            "quality_distribution": {
                "high_quality": 0,    # > 0.8
                "medium_quality": 0,  # 0.6 - 0.8
                "low_quality": 0      # < 0.6
            },
            "reward_signal_analysis": {},
            "citation_network_metrics": {},
            "temporal_analysis": {},
            "source_analysis": {}
        }
        
        if not self.provenance_traces:
            return dashboard
        
        # Analyze quality distribution
        quality_scores = []
        reward_signals_agg = defaultdict(list)
        
        for trace in self.provenance_traces.values():
            composite_quality = trace.quality_metrics.get("composite_quality", 0.5)
            quality_scores.append(composite_quality)
            
            if composite_quality > 0.8:
                dashboard["quality_distribution"]["high_quality"] += 1
            elif composite_quality > 0.6:
                dashboard["quality_distribution"]["medium_quality"] += 1
            else:
                dashboard["quality_distribution"]["low_quality"] += 1
            
            # Aggregate reward signals
            for signal, value in trace.reward_signals.items():
                reward_signals_agg[signal].append(value)
        
        # Reward signal analysis
        for signal, values in reward_signals_agg.items():
            dashboard["reward_signal_analysis"][signal] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        # Citation network metrics
        citation_counts = []
        network_sizes = []
        
        for ref_id in self.reference_mappings.keys():
            # Find original reference data
            ref_data = None
            for trace in self.provenance_traces.values():
                if ref_id in trace.reference_chain:
                    # Extract from reasoning steps (simplified)
                    for step in trace.reasoning_steps:
                        if "Citation count:" in step:
                            try:
                                count = int(step.split("Citation count:")[1].strip())
                                citation_counts.append(count)
                            except:
                                pass
                    break
        
        if citation_counts:
            dashboard["citation_network_metrics"] = {
                "total_citations": sum(citation_counts),
                "average_citations": np.mean(citation_counts),
                "citation_distribution": {
                    "highly_cited": sum(1 for c in citation_counts if c > 100),
                    "moderately_cited": sum(1 for c in citation_counts if 10 <= c <= 100),
                    "low_cited": sum(1 for c in citation_counts if c < 10)
                }
            }
        
        # Temporal analysis
        creation_years = []
        for trace in self.provenance_traces.values():
            for step in trace.reasoning_steps:
                if "Publication year:" in step:
                    try:
                        year_str = step.split("Publication year:")[1].strip()
                        if year_str != "unknown":
                            year = int(year_str[:4])
                            creation_years.append(year)
                    except:
                        pass
        
        if creation_years:
            current_year = datetime.now().year
            dashboard["temporal_analysis"] = {
                "oldest_reference": min(creation_years),
                "newest_reference": max(creation_years),
                "average_age": current_year - np.mean(creation_years),
                "age_distribution": {
                    "recent": sum(1 for y in creation_years if current_year - y <= 3),
                    "moderate": sum(1 for y in creation_years if 3 < current_year - y <= 10),
                    "old": sum(1 for y in creation_years if current_year - y > 10)
                }
            }
        
        # Source analysis
        source_counts = defaultdict(int)
        for trace in self.provenance_traces.values():
            for step in trace.reasoning_steps:
                if "ingested from" in step and "source" in step:
                    try:
                        source = step.split("ingested from")[1].split("source")[0].strip()
                        source_counts[source] += 1
                    except:
                        pass
        
        dashboard["source_analysis"] = dict(source_counts)
        
        # Overall quality metrics
        if quality_scores:
            dashboard["overall_quality"] = {
                "average_quality": np.mean(quality_scores),
                "quality_std": np.std(quality_scores),
                "quality_range": np.max(quality_scores) - np.min(quality_scores),
                "high_quality_percentage": dashboard["quality_distribution"]["high_quality"] / len(quality_scores) * 100
            }
        
        return dashboard
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics"""
        
        stats = dict(self.ingestion_stats)
        
        # Add graph statistics if available
        if self.semantic_graph:
            graph_stats = self.semantic_graph.get_graph_statistics()
            stats["graph_statistics"] = graph_stats
        
        # Add citation statistics
        stats["citation_statistics"] = {
            "total_citations": len(self.citations),
            "citation_types": {},
            "validation_status": {}
        }
        
        for citation in self.citations.values():
            citation_type = citation.citation_type
            stats["citation_statistics"]["citation_types"][citation_type] = \
                stats["citation_statistics"]["citation_types"].get(citation_type, 0) + 1
            
            validation_status = citation.validation_status
            stats["citation_statistics"]["validation_status"][validation_status] = \
                stats["citation_statistics"]["validation_status"].get(validation_status, 0) + 1
        
        # Add provenance statistics
        stats["provenance_statistics"] = {
            "total_traces": len(self.provenance_traces),
            "average_chain_length": 0,
            "average_confidence": 0,
            "average_quality_score": 0
        }
        
        if self.provenance_traces:
            total_chain_length = sum(len(trace.reference_chain) for trace in self.provenance_traces.values())
            total_confidence = sum(sum(trace.confidence_path) for trace in self.provenance_traces.values())
            total_confidence_points = sum(len(trace.confidence_path) for trace in self.provenance_traces.values())
            
            # Calculate average quality score
            quality_scores = [
                trace.quality_metrics.get("composite_quality", 0.5) 
                for trace in self.provenance_traces.values()
            ]
            
            stats["provenance_statistics"]["average_chain_length"] = total_chain_length / len(self.provenance_traces)
            stats["provenance_statistics"]["average_quality_score"] = np.mean(quality_scores)
            
            if total_confidence_points > 0:
                stats["provenance_statistics"]["average_confidence"] = total_confidence / total_confidence_points
        
        # Add quality distribution
        if self.provenance_traces:
            quality_scores = [
                trace.quality_metrics.get("composite_quality", 0.5) 
                for trace in self.provenance_traces.values()
            ]
            
            stats["quality_distribution"] = {
                "high_quality_count": sum(1 for q in quality_scores if q > 0.8),
                "medium_quality_count": sum(1 for q in quality_scores if 0.6 <= q <= 0.8),
                "low_quality_count": sum(1 for q in quality_scores if q < 0.6),
                "average_quality": np.mean(quality_scores),
                "quality_std": np.std(quality_scores)
            }
        
        return stats

# Convenience functions for backward compatibility
def load_references(path: str) -> List[Dict]:
    """Load references from JSON file"""
    pipeline = ReferenceIngestionPipeline()
    return pipeline.load_references(path)

def map_reference_to_node(ref: Dict) -> Dict:
    """Map reference to node data (backward compatibility)"""
    pipeline = ReferenceIngestionPipeline()
    return pipeline._map_reference_to_node_data(ref)

def ingest_references(graph: nx.MultiDiGraph, references: List[Dict]):
    """Ingest references into NetworkX graph (backward compatibility)"""
    pipeline = ReferenceIngestionPipeline()
    pipeline.fallback_graph = graph
    pipeline.ingest_references(references, extract_citations=False, create_provenance=False)

def get_rlhf_reward_signals(pipeline: ReferenceIngestionPipeline, 
                           research_context: Dict[str, Any]) -> Dict[str, float]:
    """
    Generate RLHF reward signals based on reference quality and research context
    
    Args:
        pipeline: Reference ingestion pipeline instance
        research_context: Context about the research query/task
        
    Returns:
        Dictionary of reward signals for RLHF training
    """
    
    reward_signals = {
        "reference_quality_reward": 0.0,
        "citation_authority_reward": 0.0,
        "source_diversity_reward": 0.0,
        "temporal_relevance_reward": 0.0,
        "network_connectivity_reward": 0.0
    }
    
    if not pipeline.provenance_traces:
        return reward_signals
    
    # Calculate reference quality reward
    quality_scores = [
        trace.quality_metrics.get("composite_quality", 0.5)
        for trace in pipeline.provenance_traces.values()
    ]
    reward_signals["reference_quality_reward"] = np.mean(quality_scores)
    
    # Calculate citation authority reward
    citation_rewards = [
        trace.reward_signals.get("citation_reward", 0.0)
        for trace in pipeline.provenance_traces.values()
    ]
    reward_signals["citation_authority_reward"] = np.mean(citation_rewards)
    
    # Calculate source diversity reward
    sources = set()
    for trace in pipeline.provenance_traces.values():
        for step in trace.reasoning_steps:
            if "ingested from" in step and "source" in step:
                try:
                    source = step.split("ingested from")[1].split("source")[0].strip()
                    sources.add(source)
                except:
                    pass
    
    # Reward diversity (more sources = higher reward)
    max_sources = 5  # Reasonable maximum for diversity
    reward_signals["source_diversity_reward"] = min(1.0, len(sources) / max_sources)
    
    # Calculate temporal relevance reward
    recency_rewards = [
        trace.reward_signals.get("recency_reward", 0.0)
        for trace in pipeline.provenance_traces.values()
    ]
    reward_signals["temporal_relevance_reward"] = np.mean(recency_rewards)
    
    # Calculate network connectivity reward
    connectivity_scores = [
        trace.quality_metrics.get("network_connectivity", 0.0)
        for trace in pipeline.provenance_traces.values()
    ]
    reward_signals["network_connectivity_reward"] = np.mean(connectivity_scores)
    
    # Apply context-specific adjustments
    query_type = research_context.get("query_type", "general")
    
    if query_type == "recent_research":
        # Boost temporal relevance for recent research queries
        reward_signals["temporal_relevance_reward"] *= 1.5
        reward_signals["temporal_relevance_reward"] = min(1.0, reward_signals["temporal_relevance_reward"])
    
    elif query_type == "foundational_concepts":
        # Boost citation authority for foundational concept queries
        reward_signals["citation_authority_reward"] *= 1.3
        reward_signals["citation_authority_reward"] = min(1.0, reward_signals["citation_authority_reward"])
    
    elif query_type == "comprehensive_survey":
        # Boost source diversity for comprehensive surveys
        reward_signals["source_diversity_reward"] *= 1.4
        reward_signals["source_diversity_reward"] = min(1.0, reward_signals["source_diversity_reward"])
    
    return reward_signals

def calculate_research_quality_score(pipeline: ReferenceIngestionPipeline,
                                   cited_references: List[str],
                                   research_output: str) -> float:
    """
    Calculate overall research quality score based on cited references
    
    Args:
        pipeline: Reference ingestion pipeline instance
        cited_references: List of reference IDs cited in research
        research_output: The research output text
        
    Returns:
        Quality score between 0 and 1
    """
    
    if not cited_references:
        return 0.1  # Low score for no citations
    
    # Get quality scores for cited references
    cited_quality_scores = []
    
    for ref_id in cited_references:
        provenance = pipeline.get_reference_provenance(ref_id)
        if provenance and provenance.get("quality_metrics"):
            quality = provenance["quality_metrics"].get("composite_quality", 0.5)
            cited_quality_scores.append(quality)
    
    if not cited_quality_scores:
        return 0.2  # Low score if no quality data available
    
    # Base quality from cited references
    base_quality = np.mean(cited_quality_scores)
    
    # Bonus for citing high-quality references
    high_quality_refs = sum(1 for q in cited_quality_scores if q > 0.8)
    quality_bonus = min(0.2, high_quality_refs / len(cited_references) * 0.2)
    
    # Bonus for diverse sources
    cited_sources = set()
    for ref_id in cited_references:
        provenance = pipeline.get_reference_provenance(ref_id)
        if provenance and provenance.get("provenance_trace"):
            for step in provenance["provenance_trace"]["reasoning_steps"]:
                if "ingested from" in step and "source" in step:
                    try:
                        source = step.split("ingested from")[1].split("source")[0].strip()
                        cited_sources.add(source)
                    except:
                        pass
    
    diversity_bonus = min(0.1, len(cited_sources) / 3.0 * 0.1)  # Up to 3 sources for full bonus
    
    # Penalty for citing only low-quality references
    low_quality_refs = sum(1 for q in cited_quality_scores if q < 0.4)
    quality_penalty = min(0.2, low_quality_refs / len(cited_references) * 0.2)
    
    # Calculate final score
    final_score = base_quality + quality_bonus + diversity_bonus - quality_penalty
    
    return max(0.0, min(1.0, final_score))
