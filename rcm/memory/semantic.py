from dataclasses import dataclass

import numpy as np


@dataclass
class MotifPrototype:
    signature: tuple
    prototype_activation: np.ndarray
    topological_pattern: dict
    expected_transition: np.ndarray
    associated_outcome: dict
    confidence: float
    usage_count: int = 1

    def to_dict(self):
        return {
            "signature": self.signature,
            "prototype_activation": self.prototype_activation.tolist(),
            "topological_pattern": self.topological_pattern,
            "expected_transition": self.expected_transition.tolist(),
            "associated_outcome": self.associated_outcome,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
        }


class SemanticMemory:
    def __init__(self):
        self.prototypes: dict[tuple, MotifPrototype] = {}

    def observe(self, manifold):
        for signature, region in manifold.regions.items():
            motif_signature = (region.specialization, len(signature))
            activation = manifold._region_activation(list(signature))
            pattern = self._topological_pattern(manifold, signature)
            outcome = self._associated_outcome(manifold, signature, region)
            if motif_signature in self.prototypes:
                prototype = self.prototypes[motif_signature]
                usage_count = prototype.usage_count + 1
                expected_transition = activation - prototype.prototype_activation
                prototype.prototype_activation = self._running_average(prototype.prototype_activation, activation, usage_count)
                prototype.expected_transition = self._running_average(prototype.expected_transition, expected_transition, usage_count)
                prototype.topological_pattern = self._merge_patterns(prototype.topological_pattern, pattern, usage_count)
                prototype.associated_outcome = self._merge_outcomes(prototype.associated_outcome, outcome, usage_count)
                prototype.confidence = ((prototype.confidence * (usage_count - 1)) + region.stability) / usage_count
                prototype.usage_count = usage_count
            else:
                self.prototypes[motif_signature] = MotifPrototype(
                    signature=motif_signature,
                    prototype_activation=activation.copy(),
                    topological_pattern=pattern,
                    expected_transition=np.zeros_like(activation),
                    associated_outcome=outcome,
                    confidence=region.stability,
                    usage_count=1,
                )

    def snapshot(self):
        return {
            signature: prototype.to_dict()
            for signature, prototype in self.prototypes.items()
        }

    def _topological_pattern(self, manifold, signature):
        internal_strengths = []
        external_strengths = []
        node_set = set(signature)
        for (source, target), edge in manifold.edges.items():
            if source in node_set and target in node_set:
                internal_strengths.append(edge.strength)
            elif source in node_set or target in node_set:
                external_strengths.append(edge.strength)
        return {
            "nodes": tuple(int(node_id) for node_id in signature),
            "node_count": len(signature),
            "mean_internal_strength": float(np.mean(internal_strengths)) if internal_strengths else 0.0,
            "mean_external_strength": float(np.mean(external_strengths)) if external_strengths else 0.0,
        }

    def _associated_outcome(self, manifold, signature, region):
        energies = [manifold.nodes[node_id].energy for node_id in signature]
        confidences = [manifold.nodes[node_id].confidence for node_id in signature]
        return {
            "stability": region.stability,
            "mean_energy": float(np.mean(energies)) if energies else 0.0,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "child_manifolds": len(manifold.child_manifolds),
        }

    def _running_average(self, previous, current, usage_count):
        return ((previous * (usage_count - 1)) + current) / usage_count

    def _merge_patterns(self, previous, current, usage_count):
        merged = dict(previous)
        merged["nodes"] = current["nodes"]
        merged["node_count"] = current["node_count"]
        for key in ("mean_internal_strength", "mean_external_strength"):
            merged[key] = ((previous[key] * (usage_count - 1)) + current[key]) / usage_count
        return merged

    def _merge_outcomes(self, previous, current, usage_count):
        merged = {}
        for key in current:
            merged[key] = ((previous[key] * (usage_count - 1)) + current[key]) / usage_count
        return merged