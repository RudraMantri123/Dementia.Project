"""Graph-based knowledge representation using NetworkX."""

from typing import Dict, Any, List, Optional, Tuple, Set
import networkx as nx
import json
import numpy as np
from collections import defaultdict

from src.database.models import KnowledgeGraphNode, KnowledgeGraphEdge
from src.database.connection import get_db_manager


class GraphKnowledgeBase:
    """Graph-based knowledge representation for medical knowledge."""

    def __init__(self):
        """Initialize graph knowledge base."""
        self.graph = nx.MultiDiGraph()
        self.db_manager = get_db_manager()
        self._load_from_db()

    def add_concept(
        self,
        concept_id: str,
        concept_type: str,
        name: str,
        description: str = "",
        properties: Dict[str, Any] = None,
        source: str = None
    ) -> bool:
        """
        Add a concept node to the knowledge graph.

        Args:
            concept_id: Unique concept identifier
            concept_type: Type of concept (symptom, treatment, medication, etc.)
            name: Concept name
            description: Detailed description
            properties: Additional properties
            source: Source of information

        Returns:
            Success status
        """
        with self.db_manager.get_session() as session:
            # Check if node exists
            existing = session.query(KnowledgeGraphNode).filter_by(
                node_id=concept_id
            ).first()

            if existing:
                return False

            # Create node in database
            node = KnowledgeGraphNode(
                node_id=concept_id,
                node_type=concept_type,
                name=name,
                description=description,
                properties=properties or {},
                source=source,
                confidence=1.0
            )

            session.add(node)
            session.commit()

            # Add to in-memory graph
            self.graph.add_node(
                concept_id,
                type=concept_type,
                name=name,
                description=description,
                properties=properties or {}
            )

            return True

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        properties: Dict[str, Any] = None,
        source: str = None
    ) -> bool:
        """
        Add a relationship between concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship
            weight: Relationship strength (0-1)
            properties: Additional properties
            source: Source of information

        Returns:
            Success status
        """
        with self.db_manager.get_session() as session:
            # Verify nodes exist
            source_node = session.query(KnowledgeGraphNode).filter_by(
                node_id=source_id
            ).first()
            target_node = session.query(KnowledgeGraphNode).filter_by(
                node_id=target_id
            ).first()

            if not source_node or not target_node:
                return False

            # Create edge in database
            edge = KnowledgeGraphEdge(
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_type=relationship_type,
                weight=weight,
                properties=properties or {},
                source=source,
                confidence=1.0
            )

            session.add(edge)
            session.commit()

            # Add to in-memory graph
            self.graph.add_edge(
                source_id,
                target_id,
                type=relationship_type,
                weight=weight,
                properties=properties or {}
            )

            return True

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find all paths between two concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def find_related_concepts(
        self,
        concept_id: str,
        relationship_types: List[str] = None,
        max_distance: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find concepts related to given concept.

        Args:
            concept_id: Concept ID to search from
            relationship_types: Filter by relationship types
            max_distance: Maximum graph distance

        Returns:
            List of related concepts with metadata
        """
        if concept_id not in self.graph:
            return []

        related = []

        # BFS to find related concepts
        visited = {concept_id}
        queue = [(concept_id, 0)]

        while queue:
            current_id, distance = queue.pop(0)

            if distance >= max_distance:
                continue

            # Get neighbors
            for neighbor in self.graph.neighbors(current_id):
                if neighbor not in visited:
                    # Get edge data
                    edges = self.graph.get_edge_data(current_id, neighbor)

                    for edge_data in edges.values():
                        rel_type = edge_data.get('type')

                        # Filter by relationship type if specified
                        if relationship_types and rel_type not in relationship_types:
                            continue

                        node_data = self.graph.nodes[neighbor]

                        related.append({
                            'concept_id': neighbor,
                            'name': node_data.get('name'),
                            'type': node_data.get('type'),
                            'relationship': rel_type,
                            'distance': distance + 1,
                            'weight': edge_data.get('weight', 1.0)
                        })

                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return related

    def find_treatments_for_symptom(self, symptom_id: str) -> List[Dict[str, Any]]:
        """
        Find treatments for a given symptom.

        Args:
            symptom_id: Symptom concept ID

        Returns:
            List of treatment concepts
        """
        treatments = []

        # Find direct treatments
        for neighbor in self.graph.neighbors(symptom_id):
            edges = self.graph.get_edge_data(symptom_id, neighbor)

            for edge_data in edges.values():
                if edge_data.get('type') == 'treats':
                    node_data = self.graph.nodes[neighbor]
                    treatments.append({
                        'concept_id': neighbor,
                        'name': node_data.get('name'),
                        'type': node_data.get('type'),
                        'weight': edge_data.get('weight', 1.0)
                    })

        # Find indirect treatments through conditions
        related = self.find_related_concepts(symptom_id, max_distance=2)

        for concept in related:
            if concept['relationship'] == 'treats' and concept['distance'] == 2:
                if concept not in treatments:
                    treatments.append(concept)

        return sorted(treatments, key=lambda x: x['weight'], reverse=True)

    def get_concept_info(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept.

        Args:
            concept_id: Concept ID

        Returns:
            Concept information or None
        """
        if concept_id not in self.graph:
            return None

        node_data = self.graph.nodes[concept_id]

        # Get relationships
        outgoing = []
        for target in self.graph.neighbors(concept_id):
            edges = self.graph.get_edge_data(concept_id, target)
            for edge_data in edges.values():
                target_data = self.graph.nodes[target]
                outgoing.append({
                    'target': target,
                    'target_name': target_data.get('name'),
                    'relationship': edge_data.get('type'),
                    'weight': edge_data.get('weight')
                })

        incoming = []
        for source in self.graph.predecessors(concept_id):
            edges = self.graph.get_edge_data(source, concept_id)
            for edge_data in edges.values():
                source_data = self.graph.nodes[source]
                incoming.append({
                    'source': source,
                    'source_name': source_data.get('name'),
                    'relationship': edge_data.get('type'),
                    'weight': edge_data.get('weight')
                })

        return {
            'concept_id': concept_id,
            'name': node_data.get('name'),
            'type': node_data.get('type'),
            'description': node_data.get('description'),
            'properties': node_data.get('properties', {}),
            'outgoing_relationships': outgoing,
            'incoming_relationships': incoming
        }

    def query_by_pattern(
        self,
        pattern: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Query graph using a pattern.

        Args:
            pattern: Pattern specification with node and edge constraints

        Returns:
            Matching subgraphs
        """
        results = []

        # Simple pattern matching implementation
        node_type = pattern.get('node_type')
        relationship_type = pattern.get('relationship_type')
        target_type = pattern.get('target_type')

        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]

            # Check if node matches pattern
            if node_type and node_data.get('type') != node_type:
                continue

            # Check outgoing edges
            for target in self.graph.neighbors(node_id):
                target_data = self.graph.nodes[target]

                if target_type and target_data.get('type') != target_type:
                    continue

                edges = self.graph.get_edge_data(node_id, target)

                for edge_data in edges.values():
                    if relationship_type and edge_data.get('type') != relationship_type:
                        continue

                    # Match found
                    results.append({
                        'source': node_id,
                        'source_name': node_data.get('name'),
                        'relationship': edge_data.get('type'),
                        'target': target,
                        'target_name': target_data.get('name')
                    })

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_types = defaultdict(int)
        relationship_types = defaultdict(int)

        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] += 1

        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('type', 'unknown')
            relationship_types[rel_type] += 1

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'relationship_types': dict(relationship_types),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }

    def export_to_json(self, filepath: str):
        """Export graph to JSON file."""
        data = nx.node_link_data(self.graph)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str):
        """Import graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.graph = nx.node_link_graph(data)

    def _load_from_db(self):
        """Load graph from database."""
        with self.db_manager.get_session() as session:
            # Load nodes
            nodes = session.query(KnowledgeGraphNode).all()

            for node in nodes:
                self.graph.add_node(
                    node.node_id,
                    type=node.node_type,
                    name=node.name,
                    description=node.description,
                    properties=node.properties or {}
                )

            # Load edges
            edges = session.query(KnowledgeGraphEdge).all()

            for edge in edges:
                self.graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    type=edge.relationship_type,
                    weight=edge.weight,
                    properties=edge.properties or {}
                )

    def initialize_medical_knowledge(self):
        """Initialize graph with basic medical knowledge."""
        # Add dementia-related concepts
        concepts = [
            ('alzheimers', 'condition', 'Alzheimer\'s Disease', 'Progressive neurodegenerative disease'),
            ('memory_loss', 'symptom', 'Memory Loss', 'Difficulty remembering recent events'),
            ('confusion', 'symptom', 'Confusion', 'Disorientation and confusion'),
            ('donepezil', 'medication', 'Donepezil', 'Cholinesterase inhibitor'),
            ('cognitive_training', 'treatment', 'Cognitive Training', 'Mental exercises to maintain function'),
            ('social_engagement', 'treatment', 'Social Engagement', 'Regular social interaction'),
        ]

        for concept_id, concept_type, name, description in concepts:
            self.add_concept(concept_id, concept_type, name, description)

        # Add relationships
        relationships = [
            ('alzheimers', 'memory_loss', 'causes', 0.9),
            ('alzheimers', 'confusion', 'causes', 0.8),
            ('donepezil', 'memory_loss', 'treats', 0.7),
            ('cognitive_training', 'memory_loss', 'treats', 0.6),
            ('social_engagement', 'confusion', 'treats', 0.5),
        ]

        for source, target, rel_type, weight in relationships:
            self.add_relationship(source, target, rel_type, weight)
