from rcm.experiments.specialization_demo import run_demo
from rcm.memory.recall import recall_node_history
from rcm.memory.recall import recall_recent_episode
from rcm.memory.recall import recall_semantic_motifs


def main():
    _, _, manifold, _, _ = run_demo(18, capture_frames=False)
    print("Recent episode:")
    print(recall_recent_episode(manifold))
    print("\nNode 0 history:")
    print(recall_node_history(manifold, 0))
    print("\nSemantic motifs:")
    print(recall_semantic_motifs(manifold, min_count=2))


if __name__ == "__main__":
    main()