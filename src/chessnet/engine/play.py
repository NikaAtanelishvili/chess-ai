"""A game against the engine in the terminal (``chessnet play``)."""

from __future__ import annotations

from collections.abc import Callable

import chess

from chessnet.engine.player import Player

HELP = (
    "Type moves in SAN (Nf3, exd5, O-O) or UCI (g1f3). Other commands: 'moves' lists the "
    "legal moves, 'undo' takes back your last move, 'resign' ends the game."
)


def render(board: chess.Board, orientation: chess.Color = chess.WHITE) -> str:
    """The board in ASCII, from ``orientation``'s side, with ranks and files."""
    ranks = range(7, -1, -1) if orientation == chess.WHITE else range(8)
    files = list(range(8)) if orientation == chess.WHITE else list(range(7, -1, -1))
    lines = ["  +-----------------+"]
    for rank in ranks:
        cells = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} | {' '.join(cells)} |")
    lines.append("  +-----------------+")
    lines.append("    " + " ".join(chess.FILE_NAMES[file] for file in files))
    return "\n".join(lines)


def parse_move(board: chess.Board, text: str) -> chess.Move:
    """A legal move from SAN or UCI text; ValueError if it is neither legal nor a move."""
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a move") from None
    if move not in board.legal_moves:
        raise ValueError(f"{text} is not legal here")
    return move


def result_text(board: chess.Board, human: chess.Color) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "Game abandoned."
    reason = outcome.termination.name.replace("_", " ").lower()
    if outcome.winner is None:
        return f"Draw ({reason}), {outcome.result()}."
    winner = "You win" if outcome.winner == human else "chessnet wins"
    return f"{winner} ({reason}), {outcome.result()}."


def play_game(
    engine: Player,
    human: chess.Color,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    describe: Callable[[], str] | None = None,
    board: chess.Board | None = None,
) -> chess.Board:
    """Play one game: the human types moves, the engine answers. Returns the final board.

    Draws are claimed automatically (threefold repetition, 50-move rule). ``describe``
    returns a note shown after each engine move, such as its evaluation.
    """
    board = board or chess.Board()
    write(f"You play {'White' if human == chess.WHITE else 'Black'}. {HELP}")
    while board.outcome(claim_draw=True) is None:
        if board.turn != human:
            move = engine.choose(board)
            note = f" ({describe()})" if describe else ""
            write(f"chessnet plays {board.san(move)}{note}")
            board.push(move)
            continue
        write(render(board, human))
        number = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
        try:
            text = read(f"{number} your move: ").strip()
        except EOFError:
            text = "resign"
        if text in {"resign", "quit", "exit"}:
            write("You resigned.")
            return board
        if text in {"moves", "help", "?"}:
            write(HELP)
            write("Legal moves: " + " ".join(sorted(board.san(m) for m in board.legal_moves)))
        elif text == "undo":
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
            else:
                write("Nothing to take back.")
        elif text:
            try:
                board.push(parse_move(board, text))
            except ValueError as error:
                write(f"{error}. {HELP}")
    write(render(board, human))
    write(result_text(board, human))
    return board
