"""
Enhanced Ablation Study with Task-Specific Objectives and Paired Analysis.

Core Objectives:
1. Next-state prediction error (HRM + semantics task)
2. Corrupted-pattern reconstruction (hierarchy + guidance task)
3. Recall accuracy (episodic memory + guidance task)
4. Classification accuracy (topology + specialization task)

Mechanism-Specific Tasks:
- GUIDANCE: Motif-based recall transfer
- HRM FIELD: Noisy temporal prediction with synchronization
- PLASTICITY: Cluster relationship changes requiring rewiring
- HIERARCHY: Compositional meta-region prediction

Analysis:
- Paired differences by graph/seed/scenario with 95% confidence intervals
- Continuous topology change metrics (edge strength evolution)
- Collapse incident consolidation (consecutive flags → incidents)
- Per-run collapse tracing
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.geometry.simplicial_complex import DynamicSimplicialComplex
from rcm.reproducibility import normalize_seed, seed_everything
from rcm_collapse_integration import install_collapse_attractor_support


install_collapse_attractor_support(RecursiveCognitiveManifold)


# ============================================================================
# STEP 1: Objective Functions
# ============================================================================

def next_state_prediction_error(manifold, test_steps: int = 10) -> float:
    """
    Measure prediction error on next-state task.
    Task: Given current state, predict next state under noise.
    """
    errors = []
    for node_id in list(manifold.nodes.keys())[:3]:  # Sample 3 nodes
        node = manifold.nodes[node_id]
        for _ in range(test_steps):
            current = node.local_state.copy()
            # Generate noisy input
            noisy_input = current + np.random.normal(0, 0.1, current.shape)
            manifold.step({node_id: noisy_input})
            predicted = node.local_state.copy()
            
            # Compare with actual trajectory
            manifold.step({node_id: noisy_input})
            actual = node.local_state.copy()
            
            error = np.linalg.norm(predicted - actual)
            errors.append(error)
    
    return float(np.mean(errors)) if errors else 1.0


def corrupted_pattern_reconstruction(manifold, test_steps: int = 5) -> float:
    """
    Measure reconstruction accuracy of corrupted patterns.
    Task: Given partial/noisy pattern, reconstruct clean pattern via hierarchy.
    """
    accuracies = []
    
    for _ in range(test_steps):
        # Create clean pattern
        clean_pattern = {}
        for node_id in manifold.nodes.keys():
            clean_pattern[node_id] = np.random.randn(manifold.state_dim).astype(float)
        
        # Corrupt it (50% dropout + noise)
        corrupted = {}
        for node_id, state in clean_pattern.items():
            if np.random.rand() > 0.5:
                corrupted[node_id] = state * np.random.uniform(0.3, 0.7) + np.random.normal(0, 0.15, state.shape)
            else:
                corrupted[node_id] = np.zeros(manifold.state_dim)
        
        # Run manifold to reconstruct
        for _ in range(3):
            manifold.step(corrupted)
        
        # Measure reconstruction
        total_sim = 0.0
        count = 0
        for node_id, clean_state in clean_pattern.items():
            reconstructed = manifold.nodes[node_id].local_state
            norm_c = np.linalg.norm(clean_state)
            norm_r = np.linalg.norm(reconstructed)
            if norm_c > 0 and norm_r > 0:
                similarity = np.dot(clean_state, reconstructed) / (norm_c * norm_r)
                total_sim += max(0.0, similarity)  # Clamp to [0, 1]
                count += 1
        
        if count > 0:
            accuracies.append(total_sim / count)
    
    return float(np.mean(accuracies)) if accuracies else 0.0


def recall_accuracy(manifold, test_steps: int = 5) -> float:
    """
    Measure episodic recall accuracy.
    Task: Encode patterns, then recall with partial cues.
    """
    accuracies = []
    
    for _ in range(test_steps):
        node_id = np.random.choice(list(manifold.nodes.keys()))
        
        # Encode a pattern
        target_pattern = np.random.randn(manifold.state_dim).astype(float)
        for _ in range(3):
            manifold.step({node_id: target_pattern})
        
        # Partial cue for recall
        cue = target_pattern * (np.random.rand(manifold.state_dim) > 0.3)  # ~70% of pattern
        
        # Measure how well episodic memory can predict next step
        memory = manifold.episodic_memory
        if len(memory.transitions) > 1:
            recalled = memory.retrieve(cue, node_id=node_id, top_k=2, min_similarity=0.0)
            if recalled and "next_state" in recalled:
                norm_target = np.linalg.norm(target_pattern)
                norm_recalled = np.linalg.norm(recalled["next_state"])
                if norm_target > 0 and norm_recalled > 0:
                    sim = np.dot(target_pattern, recalled["next_state"]) / (norm_target * norm_recalled)
                    accuracies.append(max(0.0, sim))
    
    return float(np.mean(accuracies)) if accuracies else 0.0


def classification_accuracy(manifold, test_steps: int = 5, n_classes: int = 4) -> float:
    """
    Measure classification via topology specialization.
    Task: Classify input patterns into regions based on learned specializations.
    """
    accuracies = []
    
    for _ in range(test_steps):
        # Create patterns that activate different state dimensions
        true_class = np.random.randint(0, n_classes)
        pattern = np.zeros(manifold.state_dim)
        pattern[true_class % manifold.state_dim] = 1.0
        pattern = pattern + np.random.normal(0, 0.1, manifold.state_dim)
        
        # Apply pattern and run
        for node_id in manifold.nodes.keys():
            manifold.step({node_id: pattern})
        
        # Predict class from region specializations
        predicted_class = None
        if manifold.regions:
            specializations = [r.specialization for r in manifold.regions.values()]
            spec_map = {"sensorimotor": 0, "associative": 1, "predictive": 2, "integrative": 3}
            predicted_class = spec_map.get(specializations[0], 0) if specializations else 0
        
        if predicted_class is not None:
            accuracy = 1.0 if predicted_class == true_class else 0.0
            accuracies.append(accuracy)
    
    return float(np.mean(accuracies)) if accuracies else 0.5


# ============================================================================
# STEP 2: Continuous Topology Metrics
# ============================================================================

def compute_topology_change(edges_history: List[Dict]) -> float:
    """
    Compute continuous topology change rate.
    Measures edge strength evolution, not just edge count.
    """
    if len(edges_history) < 2:
        return 0.0
    
    total_change = 0.0
    for i in range(1, len(edges_history)):
        prev_edges = edges_history[i-1]
        curr_edges = edges_history[i]
        
        # Changes in existing edges
        for key in set(prev_edges.keys()) & set(curr_edges.keys()):
            strength_change = abs(curr_edges[key] - prev_edges[key])
            total_change += strength_change
        
        # New/removed edges
        new_keys = set(curr_edges.keys()) - set(prev_edges.keys())
        removed_keys = set(prev_edges.keys()) - set(curr_edges.keys())
        total_change += len(new_keys) * 1.0 + len(removed_keys) * 1.0
    
    return float(total_change / (len(edges_history) - 1)) if len(edges_history) > 1 else 0.0


# ============================================================================
# STEP 3: Collapse Incident Consolidation
# ============================================================================

@dataclass
class CollapseIncident:
    """Represents a consolidated sequence of collapse events."""
    start_step: int
    end_step: int
    duration: int
    peak_energy: float
    total_events: int
    
    def to_dict(self):
        return asdict(self)


def consolidate_collapse_events(collapse_flags: List[Tuple[int, bool]], 
                                collapse_energies: List[float]) -> List[CollapseIncident]:
    """
    Consolidate consecutive collapse flags into incidents.
    """
    incidents = []
    in_incident = False
    incident_start = 0
    peak_energy = 0.0
    event_count = 0
    
    for step, (step_num, flag) in enumerate(collapse_flags):
        energy = collapse_energies[step] if step < len(collapse_energies) else 0.0
        
        if flag:
            if not in_incident:
                in_incident = True
                incident_start = step_num
                peak_energy = energy
                event_count = 0
            peak_energy = max(peak_energy, energy)
            event_count += 1
        else:
            if in_incident:
                incidents.append(CollapseIncident(
                    start_step=incident_start,
                    end_step=step_num - 1,
                    duration=step_num - incident_start,
                    peak_energy=peak_energy,
                    total_events=event_count,
                ))
                in_incident = False
    
    if in_incident:
        incidents.append(CollapseIncident(
            start_step=incident_start,
            end_step=len(collapse_flags) - 1,
            duration=len(collapse_flags) - incident_start,
            peak_energy=peak_energy,
            total_events=event_count,
        ))
    
    return incidents


# ============================================================================
# STEP 4: Mechanism-Specific Tasks
# ============================================================================

def guidance_task(manifold, n_steps: int = 15) -> Dict:
    """
    Test GUIDANCE mechanism: Motif-based recall and transfer.
    Learn a motif pattern and transfer it across regions.
    """
    results = {"task": "guidance_motif_transfer", "steps": n_steps, "success": False}
    
    try:
        # Phase 1: Learn motif in region A
        region_a_nodes = list(manifold.nodes.keys())[:3]
        motif = np.array([1.0, 0.5, 0.2, 0.1])
        
        for _ in range(5):
            manifold.step({node_id: motif for node_id in region_a_nodes})
        
        motif_count_before = len(manifold.semantic_memory.prototypes)
        
        # Phase 2: Apply motif variations to other nodes
        for step in range(n_steps):
            varied_motif = motif * (0.8 + 0.2 * np.sin(step / 3.0))
            other_nodes = list(manifold.nodes.keys())[3:]
            if other_nodes:
                manifold.step({node_id: varied_motif for node_id in other_nodes})
        
        motif_count_after = len(manifold.semantic_memory.prototypes)
        results["motifs_learned"] = motif_count_after - motif_count_before
        results["success"] = motif_count_after > motif_count_before
    except Exception as e:
        results["error"] = str(e)
    
    return results


def hrm_task(manifold, n_steps: int = 15) -> Dict:
    """
    Test HRM FIELD mechanism: Noisy temporal prediction with synchronization.
    Predict field evolution under noise.
    """
    results = {"task": "hrm_temporal_prediction", "steps": n_steps, "predictions": []}
    
    try:
        coherence_scores = []
        
        for step in range(n_steps):
            # Noisy input to challenge HRM field
            noisy_input = {}
            for node_id in manifold.nodes.keys():
                base = np.sin(step / 5.0) * np.ones(manifold.state_dim)
                noise = np.random.normal(0, 0.15, manifold.state_dim)
                noisy_input[node_id] = base + noise
            
            manifold.step(noisy_input)
            
            # Measure field coherence
            node_states = [n.local_state for n in manifold.nodes.values()]
            if node_states:
                stack = np.vstack(node_states)
                variance = float(np.mean(np.var(stack, axis=0)))
                coherence = 1.0 / (1.0 + variance)
                coherence_scores.append(coherence)
        
        results["mean_coherence"] = float(np.mean(coherence_scores))
        results["coherence_stability"] = float(np.std(coherence_scores))
        results["success"] = results["mean_coherence"] > 0.7
    except Exception as e:
        results["error"] = str(e)
    
    return results


def plasticity_task(manifold, n_steps: int = 20) -> Dict:
    """
    Test PLASTICITY mechanism: Changing cluster relationships requiring rewiring.
    Force cluster separation then fusion to trigger edge migration.
    """
    results = {"task": "plasticity_cluster_rewiring", "steps": n_steps, "rewiring_events": 0}
    
    try:
        initial_edges = len(manifold.edges)
        edge_history = [len(manifold.edges)]
        
        for step in range(n_steps):
            # Phase 1 (0-10): Separate clusters
            if step < 10:
                phase = 0
            # Phase 2 (10-20): Merge clusters
            else:
                phase = 1
            
            input_dict = {}
            nodes = sorted(manifold.nodes.keys())
            for idx, node_id in enumerate(nodes):
                if phase == 0:
                    # Separate: first half → (1,0,0,0), second half → (0,0,1,0)
                    if idx < len(nodes) // 2:
                        input_dict[node_id] = np.array([1.0, 0.0, 0.0, 0.0])
                    else:
                        input_dict[node_id] = np.array([0.0, 0.0, 1.0, 0.0])
                else:
                    # Merge: alternate → intermediate states
                    if idx % 2 == 0:
                        input_dict[node_id] = np.array([0.7, 0.0, 0.3, 0.0])
                    else:
                        input_dict[node_id] = np.array([0.3, 0.0, 0.7, 0.0])
            
            manifold.step(input_dict)
            edge_history.append(len(manifold.edges))
        
        edge_changes = sum(abs(edge_history[i] - edge_history[i-1]) for i in range(1, len(edge_history)))
        results["rewiring_events"] = edge_changes
        results["topology_change"] = compute_topology_change([
            {(s, t): e.strength for (s, t), e in manifold.edges.items()} 
            for _ in range(n_steps)
        ])
        results["success"] = edge_changes > 0
    except Exception as e:
        results["error"] = str(e)
    
    return results


def hierarchy_task(manifold, n_steps: int = 15) -> Dict:
    """
    Test HIERARCHY mechanism: Compositional meta-region prediction.
    Require child manifolds to predict aggregate patterns.
    """
    results = {"task": "hierarchy_compositional", "steps": n_steps, "child_manifolds": 0}
    
    try:
        initial_child_count = len(manifold.child_manifolds)
        
        for step in range(n_steps):
            # Create compositional pattern: different for each region
            input_dict = {}
            nodes = sorted(manifold.nodes.keys())
            mid = len(nodes) // 2
            
            for idx, node_id in enumerate(nodes):
                if idx < mid:
                    # Region A pattern
                    pattern = np.array([1.0 + 0.3*np.sin(step/5), 0.2, 0.0, 0.0])
                else:
                    # Region B pattern
                    pattern = np.array([0.0, 0.0, 1.0 + 0.3*np.sin(step/5), 0.2])
                
                input_dict[node_id] = pattern
            
            manifold.step(input_dict)
        
        final_child_count = len(manifold.child_manifolds)
        results["child_manifolds"] = final_child_count
        results["child_manifolds_created"] = final_child_count - initial_child_count
        results["success"] = final_child_count > 0
    except Exception as e:
        results["error"] = str(e)
    
    return results


# ============================================================================
# STEP 5-8: Enhanced Run with All Tasks
# ============================================================================

@dataclass
class EnhancedAblationRun:
    """Complete run metrics with objectives, tasks, and topology."""
    condition: str
    graph_id: str
    seed: int
    scenario: str
    
    # Core objectives
    pred_error: float
    recon_accuracy: float
    recall_accuracy: float
    class_accuracy: float
    
    # Topology metrics
    topology_change_rate: float
    final_edge_count: int
    edge_strength_variance: float
    
    # Collapse tracking
    collapse_incidents: List[Dict] = field(default_factory=list)
    total_incidents: int = 0
    max_incident_duration: int = 0
    
    # Task results
    guidance_success: bool = False
    hrm_coherence: float = 0.0
    plasticity_rewiring: int = 0
    hierarchy_children: int = 0
    
    # State metrics
    state_norm_mean: float = 0.0
    confidence_mean: float = 0.0
    
    def to_dict(self):
        d = asdict(self)
        d["collapse_incidents"] = [inc if isinstance(inc, dict) else asdict(inc) for inc in self.collapse_incidents]
        return d


def run_enhanced_condition(
    condition: str,
    graph_id: str,
    complex_,
    state_dim: int,
    seed: int,
    scenario: str,
    n_steps: int = 50,
):
    """Run enhanced ablation with all objectives and tasks."""
    resolved_seed = seed_everything(seed)
    
    # Create ablated manifold
    if condition == "no_semantic_guidance":
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim, resolved_seed)
        manifold._apply_semantic_guidance = lambda: None
    elif condition == "no_hrm_field":
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim, resolved_seed)
        manifold._apply_regional_fields = lambda: None
        manifold._apply_manifold_field = lambda: None
    elif condition == "fixed_topology":
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim, resolved_seed)
        original_step = manifold.step
        def step_no_rewiring(external_input=None):
            from rcm.dynamics.propagation import propagate_states
            from rcm.dynamics.synchronization import apply_synchronization
            from rcm.hierarchy.child_manifold import refresh_child_manifolds
            
            if external_input:
                for node_id, signal in external_input.items():
                    manifold.stimulate(node_id, signal)
            
            updated_states, _ = propagate_states(manifold)
            updated_states = manifold._apply_episodic_recall(updated_states)
            apply_synchronization(manifold, updated_states)
            manifold.episodic_memory.record(manifold.nodes, manifold.time_step)
            manifold.structural_memory.record(manifold.time_step, manifold.edges, [])
            manifold._apply_geometry_feedback({})
            manifold._update_regions()
            manifold._apply_semantic_guidance()
            manifold._apply_regional_fields()
            manifold._apply_manifold_field()
            manifold.semantic_memory.observe(manifold)
            refresh_child_manifolds(manifold)
            manifold._apply_child_manifold_fields()
            manifold.time_step += 1
            return manifold.snapshot()
        manifold.step = step_no_rewiring
    else:  # full
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim, resolved_seed)
    
    # Run main simulation
    state_norms = []
    confidences = []
    edges_history = []
    collapse_flags = []
    collapse_energies = []
    edge_strengths_history = []
    
    for step in range(n_steps):
        # Input based on scenario
        phase = (step // 10) % 3
        base_state = np.zeros(state_dim, dtype=float)
        base_state[phase] = 1.0
        
        nodes_list = list(manifold.nodes.keys())
        if "shuffled" in scenario:
            np.random.shuffle(nodes_list)
        
        input_dict = {}
        for idx, node_id in enumerate(nodes_list):
            signal = base_state.copy() * (0.8 + 0.2 * np.sin(step / 5.0 + idx))
            
            if "noisy" in scenario:
                signal = signal + np.random.normal(0, 0.15, state_dim)
            elif "missing" in scenario:
                signal = signal * (np.random.rand(state_dim) > 0.15)
            
            input_dict[node_id] = signal
        
        snapshot = manifold.step(input_dict)
        
        # Collect metrics
        state_norms.append(np.mean([np.linalg.norm(n.local_state) for n in manifold.nodes.values()]))
        confidences.append(np.mean([n.confidence for n in manifold.nodes.values()]))
        edges_history.append(len(manifold.edges))
        edge_strengths_history.append({
            (s, t): e.strength for (s, t), e in manifold.edges.items()
        })
        
        # Collapse tracking
        if "collapse_flag" in snapshot:
            collapse_flags.append((step, snapshot["collapse_flag"]))
            collapse_energies.append(snapshot.get("collapse_energy", 0.0))
    
    # Consolidate collapse incidents
    incidents = consolidate_collapse_events(collapse_flags, collapse_energies)
    
    # Run objective tasks
    pred_error = next_state_prediction_error(manifold, test_steps=5)
    recon_accuracy = corrupted_pattern_reconstruction(manifold, test_steps=5)
    recall_acc = recall_accuracy(manifold, test_steps=5)
    class_acc = classification_accuracy(manifold, test_steps=5)
    
    # Run mechanism-specific tasks
    guidance_result = guidance_task(manifold, n_steps=10)
    hrm_result = hrm_task(manifold, n_steps=10)
    plasticity_result = plasticity_task(manifold, n_steps=20)
    hierarchy_result = hierarchy_task(manifold, n_steps=15)
    
    # Topology metrics
    topo_change = compute_topology_change(edge_strengths_history)
    edge_strengths_flat = [s for d in edge_strengths_history for s in d.values()]
    edge_strength_var = float(np.var(edge_strengths_flat)) if edge_strengths_flat else 0.0
    
    return EnhancedAblationRun(
        condition=condition,
        graph_id=graph_id,
        seed=seed,
        scenario=scenario,
        
        pred_error=pred_error,
        recon_accuracy=recon_accuracy,
        recall_accuracy=recall_acc,
        class_accuracy=class_acc,
        
        topology_change_rate=topo_change,
        final_edge_count=len(manifold.edges),
        edge_strength_variance=edge_strength_var,
        
        collapse_incidents=[inc.to_dict() for inc in incidents],
        total_incidents=len(incidents),
        max_incident_duration=max((inc.duration for inc in incidents), default=0),
        
        guidance_success=guidance_result.get("success", False),
        hrm_coherence=hrm_result.get("mean_coherence", 0.0),
        plasticity_rewiring=plasticity_result.get("rewiring_events", 0),
        hierarchy_children=hierarchy_result.get("child_manifolds", 0),
        
        state_norm_mean=float(np.mean(state_norms)),
        confidence_mean=float(np.mean(confidences)),
    )


# ============================================================================
# STEP 9: Paired Difference Analysis
# ============================================================================

def compute_paired_differences(all_runs: List[EnhancedAblationRun]) -> Dict:
    """
    Compute paired differences for identical graph/seed/scenario across conditions.
    Report with 95% confidence intervals.
    """
    # Group by graph_id, seed, scenario
    groups = {}
    for run in all_runs:
        key = (run.graph_id, run.seed, run.scenario)
        if key not in groups:
            groups[key] = {}
        groups[key][run.condition] = run
    
    paired_diffs = {
        "pred_error": {},
        "recon_accuracy": {},
        "recall_accuracy": {},
        "class_accuracy": {},
        "hrm_coherence": {},
        "plasticity_rewiring": {},
        "hierarchy_children": {},
    }
    
    conditions = ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]
    
    for metric_name, metric_attr in [
        ("pred_error", "pred_error"),
        ("recon_accuracy", "recon_accuracy"),
        ("recall_accuracy", "recall_accuracy"),
        ("class_accuracy", "class_accuracy"),
        ("hrm_coherence", "hrm_coherence"),
        ("plasticity_rewiring", "plasticity_rewiring"),
        ("hierarchy_children", "hierarchy_children"),
    ]:
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                key = f"{cond1}_vs_{cond2}"
                diffs = []
                
                for group_runs in groups.values():
                    if cond1 in group_runs and cond2 in group_runs:
                        val1 = getattr(group_runs[cond1], metric_attr)
                        val2 = getattr(group_runs[cond2], metric_attr)
                        diffs.append(val1 - val2)
                
                if diffs:
                    diffs = np.array(diffs)
                    mean_diff = float(np.mean(diffs))
                    std_diff = float(np.std(diffs))
                    n = len(diffs)
                    se = std_diff / np.sqrt(n)
                    ci = float(1.96 * se)
                    
                    paired_diffs[metric_name][key] = {
                        "mean_diff": mean_diff,
                        "ci_lower": mean_diff - ci,
                        "ci_upper": mean_diff + ci,
                        "std_diff": std_diff,
                        "n_pairs": n,
                    }
    
    return paired_diffs


# ============================================================================
# STEP 10: Main Study and Reporting
# ============================================================================

def run_enhanced_study(
    n_seeds: int = 2,
    graph_sizes: List[int] = None,
    scenarios: List[str] = None,
    n_steps: int = 50,
):
    """Run the full enhanced ablation study."""
    if graph_sizes is None:
        graph_sizes = [6, 9]
    
    if scenarios is None:
        scenarios = ["baseline", "noisy_gaussian"]
    
    conditions = ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]
    
    all_runs = []
    
    for seed in range(n_seeds):
        for graph_size in graph_sizes:
            graph_id = f"g{graph_size}_s{seed}"
            complex_ = np.random.RandomState(42 + seed).randn(graph_size, 2)
            vertices = complex_
            faces = np.array([[i, (i+1) % graph_size, (i+2) % graph_size] for i in range(graph_size - 2)])
            complex_ = DynamicSimplicialComplex(vertices, faces)
            
            for scenario in scenarios:
                for condition in conditions:
                    print(f"Running: {graph_id}, {scenario}, {condition}")
                    try:
                        run = run_enhanced_condition(
                            condition=condition,
                            graph_id=graph_id,
                            complex_=complex_,
                            state_dim=4,
                            seed=100 + seed,
                            scenario=scenario,
                            n_steps=n_steps,
                        )
                        all_runs.append(run)
                        print(f"  ✓ pred_error={run.pred_error:.4f}, recon={run.recon_accuracy:.4f}")
                    except Exception as e:
                        print(f"  ✗ {e}")
    
    return all_runs


def save_enhanced_results(runs: List[EnhancedAblationRun], output_dir: Path = Path("artifacts/enhanced_ablation")):
    """Save comprehensive results with paired analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": [r.to_dict() for r in runs],
    }
    
    results_path = output_dir / "enhanced_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved {len(runs)} runs to {results_path}")
    
    # Paired differences
    paired_diffs = compute_paired_differences(runs)
    diffs_path = output_dir / "paired_differences.json"
    with open(diffs_path, "w") as f:
        json.dump(paired_diffs, f, indent=2)
    print(f"✓ Saved paired differences to {diffs_path}")
    
    return paired_diffs


def generate_enhanced_plots(runs: List[EnhancedAblationRun], paired_diffs: Dict, 
                           output_dir: Path = Path("artifacts/enhanced_ablation")):
    """Generate comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]
    
    # Figure 1: Objectives
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Enhanced Ablation: Core Objectives", fontsize=16, fontweight="bold")
    
    metrics = [
        ("pred_error", "Prediction Error (lower better)", axes[0, 0]),
        ("recon_accuracy", "Reconstruction Accuracy", axes[0, 1]),
        ("recall_accuracy", "Recall Accuracy", axes[1, 0]),
        ("class_accuracy", "Classification Accuracy", axes[1, 1]),
    ]
    
    for metric_name, title, ax in metrics:
        for condition in conditions:
            vals = [getattr(r, metric_name) for r in runs if r.condition == condition]
            ax.scatter([condition] * len(vals), vals, alpha=0.6, s=100)
            if vals:
                ax.hlines(np.mean(vals), condition, condition, colors='red', linestyles='--', linewidth=2)
        
        ax.set_ylabel(title.split("(")[0] if "(" in title else title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    fig.savefig(output_dir / "objectives_comparison.png", dpi=150)
    plt.close(fig)
    print("✓ Saved objectives comparison plot")
    
    # Figure 2: Mechanism Tasks
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Enhanced Ablation: Mechanism-Specific Tasks", fontsize=16, fontweight="bold")
    
    # Guidance success rate
    ax = axes[0, 0]
    for condition in conditions:
        success_rates = [1.0 if getattr(r, "guidance_success") else 0.0 for r in runs if r.condition == condition]
        rate = np.mean(success_rates) if success_rates else 0.0
        ax.bar(condition, rate, alpha=0.7)
    ax.set_ylabel("Success Rate")
    ax.set_title("GUIDANCE: Motif Transfer")
    ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=15)
    
    # HRM coherence
    ax = axes[0, 1]
    for condition in conditions:
        coherences = [getattr(r, "hrm_coherence") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(coherences), coherences, alpha=0.6, s=100)
        if coherences:
            ax.hlines(np.mean(coherences), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Mean Coherence")
    ax.set_title("HRM FIELD: Temporal Prediction")
    ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    # Plasticity rewiring
    ax = axes[1, 0]
    for condition in conditions:
        rewiring = [getattr(r, "plasticity_rewiring") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(rewiring), rewiring, alpha=0.6, s=100)
        if rewiring:
            ax.hlines(np.mean(rewiring), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Rewiring Events")
    ax.set_title("PLASTICITY: Cluster Rewiring")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    # Hierarchy children
    ax = axes[1, 1]
    for condition in conditions:
        children = [getattr(r, "hierarchy_children") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(children), children, alpha=0.6, s=100)
        if children:
            ax.hlines(np.mean(children), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Child Manifolds")
    ax.set_title("HIERARCHY: Compositional Meta-Regions")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "mechanisms_comparison.png", dpi=150)
    plt.close(fig)
    print("✓ Saved mechanisms comparison plot")
    
    # Figure 3: Topology Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Enhanced Ablation: Topology Dynamics", fontsize=16, fontweight="bold")
    
    ax = axes[0]
    for condition in conditions:
        changes = [getattr(r, "topology_change_rate") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(changes), changes, alpha=0.6, s=100)
        if changes:
            ax.hlines(np.mean(changes), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Topology Change Rate")
    ax.set_title("Continuous Edge Evolution")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for condition in conditions:
        incidents = [getattr(r, "total_incidents") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(incidents), incidents, alpha=0.6, s=100, color="red")
        if incidents:
            ax.hlines(np.mean(incidents), condition, condition, colors='darkred', linestyles='--', linewidth=2)
    ax.set_ylabel("Collapse Incidents")
    ax.set_title("Consolidated Collapse Events")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "topology_dynamics.png", dpi=150)
    plt.close(fig)
    print("✓ Saved topology dynamics plot")


def main():
    print("=" * 80)
    print("ENHANCED ABLATION STUDY WITH OBJECTIVES, TASKS, AND PAIRED ANALYSIS")
    print("=" * 80)
    
    runs = run_enhanced_study(
        n_seeds=2,
        graph_sizes=[6, 9],
        scenarios=["baseline", "noisy_gaussian"],
        n_steps=50,
    )
    
    print(f"\n✓ Completed {len(runs)} runs")
    
    paired_diffs = save_enhanced_results(runs)
    generate_enhanced_plots(runs, paired_diffs)
    
    print("\n" + "=" * 80)
    print("SUMMARY: PAIRED DIFFERENCES (95% CI)")
    print("=" * 80)
    
    for metric, comparisons in paired_diffs.items():
        if comparisons:
            print(f"\n{metric.upper()}:")
            for pair, stats_dict in comparisons.items():
                if stats_dict["n_pairs"] > 0:
                    mean_diff = stats_dict["mean_diff"]
                    ci_lower = stats_dict["ci_lower"]
                    ci_upper = stats_dict["ci_upper"]
                    print(f"  {pair}: {mean_diff:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (n={stats_dict['n_pairs']})")


if __name__ == "__main__":
    main()
