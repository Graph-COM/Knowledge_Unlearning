"""
check_subgraph_paths.py

Given:
- a subgraph file of directed triples, each line in the form:
  head_entity,relation,tail_entity,confidence
- two entities: entity1, entity2

This script checks whether all edges in the subgraph
strictly lie on some path from entity1 -> entity2,
with no extraneous edges or dead ends.

Usage example:
    python check_subgraph_paths.py --subgraph_file subgraph.txt --entity1 xxx --entity2 xxx

Assumptions:
- The subgraph text file has lines: "head,relation,tail,confidence"
- We treat the edges as directed from head -> tail.
- We'll use networkx to compute all simple paths from entity1 to entity2
  and see if each edge is used in at least one path.
"""

import argparse
import logging
import networkx as nx


def load_subgraph_triples(file_path):
    """
    Reads lines from file. Each line = "head,relation,tail,confidence"
    Returns a list of (head, relation, tail, confidence).
    """
    triples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                # Minimum: head, rel, tail
                continue
            head = parts[0].strip()
            relation = parts[1].strip()
            tail = parts[2].strip()
            conf = None
            if len(parts) > 3:
                conf = parts[3].strip()
            triples.append((head, relation, tail, conf))
    return triples


def check_subgraph_paths(subgraph_file, entity1, entity2):
    """
    1. Build a directed graph from the subgraph_file.
    2. Enumerate all simple paths from entity1 -> entity2.
    3. Gather edges used by those paths.
    4. Compare with the subgraph edges to see if any edge
       is unused (hence extraneous).

    Returns True if all edges are used in some path from
    entity1 to entity2 and there are no disconnected edges.
    Otherwise returns False.
    """
    # 1) load the subgraph
    triples = load_subgraph_triples(subgraph_file)
    if not triples:
        logging.info("No triples found in subgraph file.")
        return False

    # 2) build a directed graph
    G = nx.DiGraph()
    for (head, rel, tail, conf) in triples:
        # store the entire triple as an attribute, if needed
        G.add_edge(head, tail, relation=rel, confidence=conf)

    if entity1 not in G.nodes or entity2 not in G.nodes:
        logging.warning(f"Either {entity1} or {entity2} not in subgraph. Can't check full path usage.")
        return False

    # 3) all simple paths
    all_paths = list(nx.all_simple_paths(G, source=entity1, target=entity2))
    if not all_paths:
        logging.warning(f"No path from {entity1} to {entity2} found in subgraph.")
        return False

    # Gather all edges used in at least one path
    used_edges = set()
    for path in all_paths:
        for i in range(len(path) - 1):
            used_edges.add((path[i], path[i+1]))

    # 4) compare with subgraph edges
    subgraph_edges = set((h, t) for (h, _, t, _) in triples)

    unused_edges = subgraph_edges - used_edges

    if unused_edges:
        logging.info("Some edges in the subgraph are NOT on any path from "
                     f"{entity1} -> {entity2}: {unused_edges}")
        logging.info("Hence, the subgraph is NOT exclusively composed of valid paths.")
        return False

    logging.info(f"Success! All subgraph edges lie on a path from '{entity1}' to '{entity2}'.")
    logging.info(f"Number of distinct simple paths found: {len(all_paths)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a subgraph exclusively composes valid paths from entity1 to entity2"
    )
    parser.add_argument("--subgraph_file", required=True,
                        help="Path to the subgraph file with lines: head,relation,tail,confidence")
    parser.add_argument("--entity1", required=True, help="Starting entity")
    parser.add_argument("--entity2", required=True, help="Target entity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    result = check_subgraph_paths(args.subgraph_file, args.entity1, args.entity2)
    if result:
        logging.info("The subgraph is a perfect path subgraph from entity1 to entity2.")
    else:
        logging.info("The subgraph has extraneous edges or no valid path.")


if __name__ == "__main__":
    main()
