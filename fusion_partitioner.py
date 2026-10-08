"""Fusion partitioner for capture: torch's CapabilityBasedPartitioner, pruned.

torch's ``propose_partitions`` walks the full transitive downstream closure of
EVERY user of a partition, both when it records which partitions a new one can
reach and when it checks whether a merge would create a cycle. On a large graph
those closures are near-total and heavily overlapping, so a single
``propose_partitions`` call does O(nodes * users * closure) work: a production
graph with ~11k post-grad nodes did not finish capture in 50 minutes.

Both loops only depend on the UNION of ``{user} | downstreams(user)`` over the
users. A user that sits inside an earlier user's downstream closure contributes
nothing (its own closure is a subset of the earlier one), so visiting users in
topological order and skipping covered ones gives the same union, and the same
partitions, for a fraction of the work. The two loops marked PRUNED below are the
only algorithmic change (``skip_horizontal_fusion`` is also read with getattr so
torch versions without it still work); the rest is torch's ``propose_partitions``
verbatim (torch/fx/passes/infra/partitioner.py, BSD-3-Clause). A torch upgrade
that changes the algorithm must be re-copied: tests/test_fusion_partitioner.py
checks that the partitions match the installed torch's.
"""
from __future__ import annotations

import collections
import itertools
import logging
import operator
from collections.abc import Iterable

from torch.fx.node import _get_qualified_name, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition

logger = logging.getLogger(__name__)


class PrunedCapabilityBasedPartitioner(CapabilityBasedPartitioner):
    def propose_partitions(self) -> list[Partition]:
        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map: dict[int, set[int]] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: dict[Node, int] = {}  # mapping from node to partition_id
        partitions_by_id: dict[
            int, Partition
        ] = {}  # mapping from partition_id to partition
        nodes_order: dict[
            Node, int
        ] = {}  # mapping from nodes to reversed topological order
        partitions_order: dict[
            int, int
        ] = {}  # mapping from partition_id to minimum topo order of nodes in partition
        partition_users: dict[
            int, set[Node]
        ] = {}  # mapping from partition_id to partition users
        new_partition_id = itertools.count()
        topo_order = {n: i for i, n in enumerate(self.graph_module.graph.nodes)}

        def downstream_partitions(
            user_nodes: Iterable[Node], source_ids: Iterable[int]
        ) -> set[int]:
            source_ids_set = set(source_ids)
            downstream_partition_ids: set[int] = set()
            # PRUNED: visit users in topological order and skip any user already in
            # an earlier user's downstream closure (its closure is a subset), and
            # union each partition's reachability set once.
            covered: set[Node] = set()
            expanded: set[int] = set()
            for user_node in sorted(user_nodes, key=topo_order.__getitem__):
                if user_node in covered:
                    continue
                downstreams = self.dependency_viewer.downstreams_of(user_node)
                covered |= downstreams
                for path_node in itertools.chain((user_node,), downstreams):
                    target_id = assignment.get(path_node)
                    if target_id is None or target_id in expanded:
                        continue
                    expanded.add(target_id)
                    downstream_partition_ids.add(target_id)
                    downstream_partition_ids.update(partition_map[target_id])

            downstream_partition_ids.difference_update(source_ids_set)
            return downstream_partition_ids

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int) -> tuple[int, bool]:
            # merged_nodes is the union of nodes in two partition to-be-merged
            self_nodes = partitions_by_id[self_id].nodes
            other_nodes = partitions_by_id[other_id].nodes

            def dfs_iter_find_cycle(all_user_nodes: set[Node]) -> bool:
                # PRUNED: same user skipping as downstream_partitions. A partition's
                # cycle verdict doesn't depend on which user reached it, so
                # visited_partition_ids is shared across users.
                covered: set[Node] = set()
                visited_partition_ids = set()
                for user_node in sorted(all_user_nodes, key=topo_order.__getitem__):
                    if user_node in covered:
                        continue
                    downstreams = self.dependency_viewer.downstreams_of(user_node)
                    covered |= downstreams

                    for path_node in itertools.chain((user_node,), downstreams):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in self_nodes or path_node in other_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the partitions that are traversed in
                        # the dfs path and checking if they are in the merged_nodes list.
                        if path_node in assignment:
                            partition_id = assignment[path_node]
                            # If the partition id has already been visited then we know that it doesn't
                            # form a cycle with the current partitions being merged.
                            if partition_id in visited_partition_ids:
                                continue
                            p_map = partition_map[partition_id]
                            if self_id in p_map or other_id in p_map:
                                return True

                            visited_partition_ids.add(partition_id)

                return False

            # find new partition users if merge.
            all_user_nodes = partition_users[self_id] | partition_users[other_id]
            all_user_nodes.difference_update(other_nodes, self_nodes)

            # check if merge would create cyclic dependency.
            if dfs_iter_find_cycle(all_user_nodes):
                # return false indicating cyclic dependency found and
                # merge is aborted
                return self_id, False

            # merge the smaller partition into the larger.
            merge_id, removed_id = self_id, other_id
            if len(self_nodes) < len(other_nodes):
                merge_id, removed_id = removed_id, merge_id
            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[merge_id].nodes.update(partitions_by_id[removed_id].nodes)
            # updating assignment map
            for node in partitions_by_id[removed_id].nodes:
                assignment[node] = merge_id
            # delete other partition
            del partitions_by_id[removed_id]

            partitions_order[merge_id] = min(
                partitions_order[merge_id], partitions_order[removed_id]
            )
            del partitions_order[removed_id]

            partition_map[merge_id] = partition_map[merge_id].union(
                partition_map[removed_id]
            )
            del partition_map[removed_id]

            partition_users[merge_id] = all_user_nodes
            del partition_users[removed_id]

            partition_map[merge_id].update(
                downstream_partitions(all_user_nodes, (merge_id, removed_id))
            )
            partition_map[merge_id].discard(merge_id)
            partition_map[merge_id].discard(removed_id)

            return merge_id, True

        def merge_single_node(
            node: Node, node_order: int | None, id: int | None
        ) -> None:
            def _update_partition_map(node: Node, id: int) -> None:
                # Update reachability through both assigned and unsupported users.
                partition_map[id].update(downstream_partitions(node.users, (id,)))

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                if node_order is None:
                    raise AssertionError("node_order is required for new partitions")
                partitions_by_id[id] = Partition(
                    id=id, nodes=[node], node_orders=[node_order]
                )
                partition_users[id] = set(node.users)
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node, node_order)

        logger.debug("Proposing partitions...")

        for node_order, node in enumerate(reversed(self.graph_module.graph.nodes)):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: dict[int, None] = {}
            created_partition = False

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # When skip_horizontal_fusion is True, only merge the newly-created
            # partition for this supported node with partitions for its direct users.
            # This preserves data-dependent vertical fusion while avoiding fusion of
            # independent consumer partitions around an unsupported node.
            if self._is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                nodes_order[node] = partition_id
                partitions_order[partition_id] = partition_id
                merge_single_node(node, node_order, partition_id)
                merge_candidates[partition_id] = None
                created_partition = True

            if getattr(self, "skip_horizontal_fusion", False):
                if created_partition:
                    for user in node.users:
                        if user in assignment:
                            merge_candidates[assignment[user]] = None
            else:
                # merge all possible partitions
                for partition_id, _ in sorted(
                    partitions_order.items(), key=operator.itemgetter(1)
                ):
                    merge_candidates[partition_id] = None

            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partitions if it doesn't create cyclic dependency
                    # in the graph, otherwise, this is a no-op
                    self_id, _ = maybe_merge_partition(self_id, other_id)

        # sort partition nodes based on descending node order
        for partition in partitions_by_id.values():
            partition.nodes = dict(
                sorted(
                    partition.nodes.items(), key=operator.itemgetter(1), reverse=True
                )
            )

        # post processing to re-assign "getitem" nodes into upstream partition
        # Run iteratively until no more changes, to handle nested getitem chains
        # (e.g., getitem_619 = getitem_618[0] where getitem_618 = with_effects_167[1])
        logger.debug("Reassigning getitem nodes to its producer node's partition...")
        while True:
            nodes_reassignment: dict[Node, int] = {}
            for node in self.graph_module.graph.nodes:
                is_tuple_output = True
                for user in node.users:
                    if (
                        user.op != "call_function"
                        or _get_qualified_name(user.target) != "_operator.getitem"
                    ):  # type: ignore[arg-type]
                        is_tuple_output = False
                        break

                # node has tuple outputs, re-assign all following getitem node into node's partition
                if is_tuple_output:
                    id = assignment.get(node)  # type: ignore[arg-type]
                    for user in node.users:
                        if assignment.get(user) != id:  # type: ignore[arg-type]
                            nodes_reassignment[user] = id  # type: ignore[assignment]

            # no more re-assignments
            if not nodes_reassignment:
                break

            for node, id in nodes_reassignment.items():
                merge_single_node(node, None, id)

        # filter out single node partitions
        if not self.allows_single_node_partition:
            logger.debug("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: list[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        if not callable(node.target):
                            raise AssertionError(
                                f"Expected callable target, got {type(node.target)}"
                            )
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if (
                            _get_qualified_name(node.target)
                            in self.allowed_single_node_partition_ops
                        ):
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logger.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logger.debug(
                "partition #%s: %s", id, [node.name for node in partition.nodes]
            )

        return [
            partition for partition in partitions_by_id.values() if partition.size() > 0
        ]

