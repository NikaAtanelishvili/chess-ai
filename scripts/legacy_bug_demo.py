"""Demonstrate defects in the legacy engine (tag ``legacy-v0``).

Throwaway script backing the evidence in ``docs/AUDIT.md``; the bug-class
labels printed below are defined in that file. It will be removed together
with the legacy code.

Run from the repository root:

    uv run python scripts/legacy_bug_demo.py

The legacy modules live in ``legacy/`` but import each other as ``src.*``;
this script aliases ``src`` to ``legacy/`` so they load unmodified. The legacy
``utils.py`` runs a PGN filter at import time (reading and rewriting PGN files
in the current directory). This script loads that module with the top-level
call stripped, so running it touches no files.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import types
from pathlib import Path

sys.dont_write_bytecode = True  # keep src/__pycache__ untouched

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"


def _alias_legacy_package() -> None:
    """Make ``import src.<module>`` resolve to ``legacy/<module>.py``."""
    package = types.ModuleType("src")
    package.__path__ = [str(LEGACY)]
    sys.modules["src"] = package


def _load_legacy_utils() -> types.ModuleType:
    """Import legacy/utils.py without its import-time side effect."""
    path = LEGACY / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))
    ]
    module = types.ModuleType("src.utils")
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


_alias_legacy_package()
sys.modules["src.utils"] = _load_legacy_utils()

import chess  # noqa: E402
import numpy as np  # noqa: E402

from src import cnn_dataset, mcts2  # noqa: E402
from src.utils import board_to_tensor  # noqa: E402


def header(title: str) -> None:
    print(f"\n=== {title}")


def encode_single(board: chess.Board):
    """Legacy encoder with no history (4 empty history slots)."""
    return board_to_tensor([None, None, None, None, board])


class UniformStub:
    """Stand-in network: uniform priors over all 4672 indices, value 0."""

    def evaluate(self, _tensor):
        return np.full(mcts2.POLICY_SIZE, 1.0 / mcts2.POLICY_SIZE, np.float32), 0.0


def demo_b1() -> None:
    header("B1  board not mirrored for Black (training encoder)")
    black_to_move = chess.Board()
    black_to_move.push_san("e4")
    mirrored = black_to_move.mirror()  # same position, colours swapped, White to move
    a, b = encode_single(black_to_move), encode_single(mirrored)
    print(f"position: {black_to_move.fen()}")
    print(f"mirror  : {mirrored.fen()}")
    print(f"encodings identical: {bool((a == b).all())}  (correct encoder: True)")
    print(f"differing cells: {int((a != b).sum())} of {a.numel()}")
    print(f"side-to-move plane value: original {a[12, 0, 0]:.0f}, mirror {b[12, 0, 0]:.0f}")


def demo_b2() -> None:
    header("B2  move index mirrored but board not (inside MCTS, AI = White)")
    # A depth-1 node in the White AI's tree: Black to move, board in real orientation.
    node_board = chess.Board()
    node_board.push_san("e4")
    net_input_board = mcts2.canonicalize_board(node_board, chess.WHITE)
    print(f"board fed to the net      : {net_input_board.fen()}  (not mirrored, turn forced to w)")
    black_reply = chess.Move.from_uci("e7e5")
    idx = mcts2.move_to_index(black_reply, node_board)
    print(f"index of Black's e7e5     : {idx}")
    print(f"index of White's e2e4     : {mcts2.move_to_index(chess.Move.from_uci('e2e4'), chess.Board())}")
    from_sq = idx // len(mcts2.ACTIONS)
    print(
        f"index's from-square       : {chess.square_name(from_sq)}, which on the board "
        f"the net sees holds {net_input_board.piece_at(from_sq)}; the moving pawn is on "
        f"e7 there, so policy and input disagree"
    )


def demo_b4() -> None:
    header("B4  MCTS backup sign is one ply off (stub evaluator: uniform priors, value 0)")
    cases = [
        (chess.WHITE, "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
        (chess.BLACK, "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),
    ]
    for color, fen, mate in cases:
        board = chess.Board(fen)
        ai = mcts2.ChessAI(UniformStub(), color=color, sims=300, c_puct=1.5)
        chosen = ai.select_move(board)
        canon = mcts2.canonicalize_board(board, color)
        mate_canon = mcts2.canonicalize_move(chess.Move.from_uci(mate), color)
        root = ai.tt.get(chess.polyglot.zobrist_hash(canon))
        child = root.children[mcts2.move_to_index(mate_canon, canon)]
        side = "White" if color == chess.WHITE else "Black"
        print(
            f"{side}: mate-in-1 {mate} -> visits {child.N}, Q {child.Q:+.2f}; "
            f"engine plays {chosen.uci()} (mate found: {chosen.uci() == mate})"
        )


def demo_b6() -> None:
    header("B6  decoder adds a queen promotion to non-pawn moves from the 7th rank")
    board = chess.Board("4k3/R7/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("a7a8")
    decoded = mcts2.index_to_move(mcts2.move_to_index(move, board), board.turn)
    print(f"rook move {move.uci()} -> index -> {decoded.uci()}; decoded move legal: {decoded in board.legal_moves}")


def demo_b9() -> None:
    header("B9  priors are not renormalised over legal moves")
    board = chess.Board()
    policy, _ = UniformStub().evaluate(None)
    legal_mass = sum(policy[mcts2.move_to_index(m, board)] for m in board.legal_moves)
    print(f"prior mass on the 20 legal moves at the start position: {legal_mass:.4f} (should be 1.0)")


def demo_history_order() -> None:
    header("Extra  MCTS history is built in reverse order")
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)
    ai = mcts2.ChessAI(UniformStub(), color=chess.BLACK, sims=1)
    history = ai._create_canonical_history(board)
    print(f"real position         : {board.fen()}")
    print(f"encoded as 'current'  : {history[-1].fen()}  (canonical frame)")
    print(f"is the start position : {history[-1].board_fen() == chess.Board().board_fen()}")


PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/demo0001"]
[White "w"]
[Black "b"]
[Result "0-1"]

1. e4 { [%eval 0.17] } 1... e5 { [%eval 0.25] } 2. Qh5 { [%eval -0.4] } 0-1
"""


def demo_dataset() -> None:
    header("Extra / B3  training dataset: crashes as written; targets misaligned")
    with tempfile.TemporaryDirectory() as tmp:
        pgn = Path(tmp) / "demo.pgn"
        pgn.write_text(PGN, encoding="utf-8")
        try:
            cnn_dataset.LichessPGNDataset([str(pgn)])
            print("dataset built (unexpected)")
        except TypeError as exc:
            print(f"LichessPGNDataset raises TypeError: {exc}")

        # Patch the index call to record the raw move, to show what each sample pairs up.
        original = cnn_dataset.move_to_index
        cnn_dataset.move_to_index = lambda move: move
        try:
            dataset = cnn_dataset.LichessPGNDataset([str(pgn)])
        finally:
            cnn_dataset.move_to_index = original
    print("game result: 0-1 (Black won); the result is never read")
    for history, target, value in dataset.samples:
        current = history[-1]
        mover = "White" if current.turn == chess.WHITE else "Black"
        print(
            f"input: {mover} to move | policy target {target.uci()} "
            f"(legal here: {target in current.legal_moves}) | value target {value:+.3f} (White's view)"
        )


def main() -> None:
    demo_b1()
    demo_b2()
    demo_b4()
    demo_b6()
    demo_b9()
    demo_history_order()
    demo_dataset()


if __name__ == "__main__":
    main()
