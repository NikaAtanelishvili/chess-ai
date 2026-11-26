"""Monte Carlo tree search: PUCT with batched leaf evaluation, virtual loss and tree reuse.

Values are expected outcomes in ``[-1, 1]``. A node's ``value_sum`` is seen from the player
who made the move into it, so a parent compares its children's mean values directly, and
:func:`backup` negates the value at every ply.

Terminal positions are scored without the network (:func:`terminal_value`): checkmate is -1
for the side to move; stalemate, insufficient material, threefold repetition and the
50-move rule are draws. Repetitions are counted over the real game: the search pushes its
moves onto a copy of the game board that keeps the whole move stack. The root itself is
always searched, even where a draw could be claimed.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import chess

from chessnet.search.evaluator import Evaluator


@dataclass(frozen=True)
class SearchConfig:
    """PUCT settings."""

    c_puct: float = 2.0  # weight of the prior-driven exploration term
    fpu_reduction: float = 0.2  # unvisited children score the parent's value minus this
    batch_size: int = 16  # leaves evaluated together in one network call

    def __post_init__(self) -> None:
        if self.c_puct <= 0 or self.fpu_reduction < 0 or self.batch_size < 1:
            raise ValueError(f"invalid search settings: {self}")


class Node:
    """A position in the search tree, reached from its parent by one move."""

    __slots__ = ("children", "pending", "prior", "terminal", "value_sum", "virtual", "visits")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0  # for the player who made the move into this node
        self.virtual = 0  # selections through this node still waiting for an evaluation
        self.children: dict[chess.Move, Node] | None = None  # None until expanded
        self.terminal: float | None = None  # the game is over here: value for the side to move
        self.pending = False  # waiting for the network call in progress

    @property
    def q(self) -> float:
        """Mean value for the player who made the move into this node; 0 if unvisited."""
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one search, seen from the side to move at the root."""

    move: chess.Move  # the most visited root move
    visits: dict[chess.Move, int]  # visits of every root move
    value: float  # expected outcome for the side to move
    move_value: float  # expected outcome for the side to move after ``move``
    mate_in_one: bool  # ``move`` checkmates
    pv: list[chess.Move]  # principal variation: the most visited line
    playouts: int  # playouts in this search (each adds one visit to the root)
    nodes: int  # root visits, including those of a reused tree
    seconds: float

    @property
    def nodes_per_second(self) -> float:
        return self.playouts / self.seconds if self.seconds > 0 else 0.0


def terminal_value(board: chess.Board) -> float | None:
    """The result for the side to move if the game is over, counting claimable draws."""
    if not any(board.generate_legal_moves()):
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
        return 0.0
    return None


def backup(path: Sequence[Node], value: float) -> None:
    """Add a leaf's ``value`` (for its side to move) to every node on ``path``, root first.

    The sign flips at every ply, because each node keeps the value for the player who made
    the move into it.
    """
    for node in reversed(path):
        value = -value
        node.visits += 1
        node.value_sum += value


def most_visited(node: Node) -> chess.Move:
    """The child move with the most visits (ties: better mean value, then larger prior)."""
    children = node.children
    if not children:
        raise ValueError("the node has no children")
    return max(
        children, key=lambda move: (children[move].visits, children[move].q, children[move].prior)
    )


def sample_move(
    visits: dict[chess.Move, int], temperature: float, rng: random.Random
) -> chess.Move:
    """A move drawn with probability proportional to ``visits ** (1 / temperature)``."""
    most = max(visits.values())
    if temperature <= 0 or most == 0:
        return max(visits, key=visits.__getitem__)
    moves = [move for move, count in visits.items() if count > 0]
    weights = [(visits[move] / most) ** (1 / temperature) for move in moves]
    return rng.choices(moves, weights)[0]


class MCTS:
    """PUCT search over positions, reusing the subtree of the moves actually played.

    Each call of :meth:`search` adds ``simulations`` playouts (or runs for ``movetime``
    seconds) to the tree of the given board. If the board continues the game of the previous
    search, the tree below the moves played since is kept.
    """

    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.root: Node | None = None
        self._root_board: chess.Board | None = None

    def search(
        self,
        board: chess.Board,
        *,
        simulations: int | None = None,
        movetime: float | None = None,
        stop: threading.Event | None = None,
        max_nodes: int | None = None,
        on_progress: Callable[[SearchResult], None] | None = None,
        progress_interval: float = 1.0,
    ) -> SearchResult:
        """Search ``board`` until the first of: ``simulations`` playouts, ``movetime``
        seconds, ``stop`` being set, or ``max_nodes`` root visits (which bounds memory).

        Give at least one of ``simulations``, ``movetime`` and ``stop``. The first playout
        always runs, so there is a move to play even if ``stop`` is already set.
        ``on_progress`` receives interim results every ``progress_interval`` seconds.
        """
        if simulations is None and movetime is None and stop is None:
            raise ValueError("give simulations, movetime or a stop event")
        if simulations is not None and simulations < 1:
            raise ValueError(f"simulations must be positive, not {simulations}")
        if not any(board.generate_legal_moves()):
            raise ValueError("no legal moves: the game is over")
        root = self._set_root(board)
        game = board.copy()  # keeps the move stack, so repetitions include the real game
        started = last_report = time.monotonic()
        playouts = 0
        while True:
            limit = self.config.batch_size
            if simulations is not None:
                limit = min(limit, simulations - playouts)
            playouts += self._run_batch(root, game, limit)
            now = time.monotonic()
            if (
                (simulations is not None and playouts >= simulations)
                or (movetime is not None and now - started >= movetime)
                or (stop is not None and stop.is_set())
                or (max_nodes is not None and root.visits >= max_nodes)
            ):
                break
            if on_progress is not None and now - last_report >= progress_interval:
                on_progress(self._result(root, playouts, now - started))
                last_report = now
        return self._result(root, playouts, time.monotonic() - started)

    def principal_variation(self) -> list[chess.Move]:
        """The most visited line from the root."""
        line: list[chess.Move] = []
        node = self.root
        while node is not None and node.children and node.visits > 1:
            move = most_visited(node)
            child = node.children[move]
            if child.visits == 0:
                break
            line.append(move)
            node = child
        return line

    def _set_root(self, board: chess.Board) -> Node:
        node = self._reusable_node(board) or Node(prior=1.0)
        node.terminal = None  # the root is searched even where a draw could be claimed
        self.root = node
        self._root_board = board.copy()
        return node

    def _reusable_node(self, board: chess.Board) -> Node | None:
        """The node of ``board`` in the current tree, if ``board`` continues its game."""
        old = self._root_board
        node = self.root
        if node is None or old is None:
            return None
        played = len(old.move_stack)
        if board.move_stack[:played] != old.move_stack or board.root() != old.root():
            return None
        for move in board.move_stack[played:]:
            if node.children is None or move not in node.children:
                return None
            node = node.children[move]
        return node

    def _select_child(self, node: Node) -> tuple[chess.Move, Node]:
        children = node.children
        assert children, "only expanded, non-terminal nodes are selected through"
        explore = self.config.c_puct * math.sqrt(max(node.visits + node.virtual, 1))
        unvisited = -node.q - self.config.fpu_reduction  # the parent's value for its mover
        best_score, best = -math.inf, next(iter(children.items()))
        for move, child in children.items():
            visits = child.visits + child.virtual
            # Each pending selection counts as a loss for the player who chose the child.
            mean = (child.value_sum - child.virtual) / visits if visits else unvisited
            score = mean + explore * child.prior / (1 + visits)
            if score > best_score:
                best_score, best = score, (move, child)
        return best

    def _run_batch(self, root: Node, game: chess.Board, limit: int) -> int:
        """Up to ``limit`` playouts that share one network call; returns how many finished."""
        leaves: list[tuple[list[Node], chess.Board]] = []
        finished = 0
        for _ in range(limit):
            node, path, pushed = root, [root], 0
            while node.children is not None and node.terminal is None:
                move, node = self._select_child(node)
                game.push(move)
                path.append(node)
                pushed += 1
            collided = node.pending
            if node.terminal is not None:
                backup(path, node.terminal)
                finished += 1
            elif not collided:
                value = terminal_value(game) if pushed else None
                if value is not None:
                    node.terminal = value
                    backup(path, value)
                    finished += 1
                else:
                    node.pending = True
                    for visited in path:
                        visited.virtual += 1
                    leaves.append((path, game.copy(stack=False)))
            for _ in range(pushed):
                game.pop()
            if collided:  # the leaf is already waiting in this batch
                break
        if leaves:
            evaluations = self.evaluator.evaluate([board for _, board in leaves])
            for (path, _), evaluation in zip(leaves, evaluations, strict=True):
                leaf = path[-1]
                leaf.children = {move: Node(prior) for move, prior in evaluation.priors.items()}
                leaf.pending = False
                for visited in path:
                    visited.virtual -= 1
                backup(path, evaluation.value)
                finished += 1
        return finished

    def _result(self, root: Node, playouts: int, seconds: float) -> SearchResult:
        children = root.children
        assert children, "the root is expanded by its first playout"
        move = most_visited(root)
        chosen = children[move]
        return SearchResult(
            move=move,
            visits={move: child.visits for move, child in children.items()},
            value=-root.q,
            move_value=chosen.q if chosen.visits else -root.q,
            mate_in_one=chosen.terminal == -1.0,
            pv=self.principal_variation(),
            playouts=playouts,
            nodes=root.visits,
            seconds=seconds,
        )
