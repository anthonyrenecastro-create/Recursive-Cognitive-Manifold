from collections import deque

import numpy as np


class EpisodicMemory:
    def __init__(self, maxlen: int = 64):
        self.events = deque(maxlen=maxlen)
        self.transitions = deque(maxlen=maxlen * 16)

    def record(self, nodes, time_step: int):
        activations = {node_id: node.local_state.copy() for node_id, node in nodes.items()}
        energy = {node_id: node.energy for node_id, node in nodes.items()}
        confidence = {node_id: node.confidence for node_id, node in nodes.items()}

        if self.events:
            previous_episode = self.events[-1]
            previous_episode["next_activations"] = activations
            previous_episode["outcome"] = {
                "energy": energy,
                "confidence": confidence,
            }
            for node_id, current_state in previous_episode["activations"].items():
                if node_id not in activations:
                    continue
                self.transitions.append(
                    {
                        "node_id": node_id,
                        "time_step": previous_episode["time_step"],
                        "current_state": current_state.copy(),
                        "next_state": activations[node_id].copy(),
                        "outcome": {
                            "energy": energy[node_id],
                            "confidence": confidence[node_id],
                        },
                    }
                )

        episode = {
            "time_step": time_step,
            "activations": activations,
            "energy": energy,
            "confidence": confidence,
            "next_activations": None,
            "outcome": None,
        }
        self.events.append(episode)
        for node in nodes.values():
            node.long_term_memory.append((time_step, node.local_state.copy(), node.energy))
        return episode

    def retrieve(self, current_state, node_id: int | None = None):
        best_match = None
        best_similarity = -1.0
        for transition in self.transitions:
            if node_id is not None and transition["node_id"] != node_id:
                continue
            similarity = self._state_similarity(current_state, transition["current_state"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = transition

        if best_match is None:
            return None

        recalled = dict(best_match)
        recalled["similarity"] = best_similarity
        return recalled

    def _state_similarity(self, left, right) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))