import math
import time
import chess
import requests

INF = 10**9

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # king value is handled via checkmate detection
}

def material_score(board: chess.Board) -> int:
    """White-centric centipawn score (white minus black)."""
    score = 0
    for p, v in PIECE_VALUE.items():
        score += len(board.pieces(p, chess.WHITE)) * v
        score -= len(board.pieces(p, chess.BLACK)) * v
    return score

def mobility_score(board: chess.Board) -> int:
    # More legal moves = slightly better. Tiny weight to avoid overpowering material.
    moves = board.legal_moves.count()
    # White to move: mobility benefits white; black to move: mobility hurts white a bit.
    return (moves if board.turn == chess.WHITE else -moves) * 2

def evaluate(board: chess.Board) -> int:
    """Static evaluation from White's perspective (centipawns)."""
    if board.is_checkmate():
        # If it's White to move and checkmated, white lost.
        return -INF if board.turn == chess.WHITE else INF
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0
    return material_score(board) + mobility_score(board)

def order_moves(board: chess.Board):
    """Simple move ordering: captures & checks first."""
    def key(m: chess.Move):
        cap = board.is_capture(m)
        chk = board.gives_check(m)
        # MVV-LVA heuristic for captures
        victim = board.piece_at(m.to_square).piece_type if cap and board.piece_at(m.to_square) else 0
        attacker = board.piece_at(m.from_square).piece_type if board.piece_at(m.from_square) else 0
        mvv_lva = PIECE_VALUE.get(victim, 0) - PIECE_VALUE.get(attacker, 0) // 10
        return (cap, chk, mvv_lva)
    return sorted(board.legal_moves, key=key, reverse=True)

def quiescence(board: chess.Board, alpha: int, beta: int, color: int) -> int:
    """Captures-only search at leaves."""
    stand_pat = color * evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Consider only captures to stabilize evaluation
    for move in board.legal_moves:
        if not board.is_capture(move) and not board.gives_check(move):
            continue
        board.push(move)
        val = -quiescence(board, -beta, -alpha, -color)
        board.pop()
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    return alpha

def negamax(board: chess.Board, depth: int, alpha: int, beta: int, color: int,
            leaf_engine=None, leaf_ms: int = 0) -> tuple[int, chess.Move | None]:
    """Negamax with alpha-beta. color = +1 (white POV) if side to move is white, else -1."""
    if depth == 0 or board.is_game_over():
        # Optional: consult engine briefly at leaves for stronger play (still your agent).
        if leaf_engine:
            try:
                import chess.engine
                info = leaf_engine.analyse(board, chess.engine.Limit(time=leaf_ms / 1000.0))
                score = info["score"].pov(chess.WHITE).score(mate_score=100000)
                return color * (score if score is not None else evaluate(board)), None
            except Exception:
                pass
        return quiescence(board, alpha, beta, color), None

    best_val = -INF
    best_move = None

    for move in order_moves(board):
        board.push(move)
        val, _ = negamax(board, depth - 1, -beta, -alpha, -color, leaf_engine, leaf_ms)
        val = -val
        board.pop()

        if val > best_val:
            best_val = val
            best_move = move
        if best_val > alpha:
            alpha = best_val
        if alpha >= beta:
            break  # beta cutoff

    return best_val, best_move

def choose_move_minimax(board: chess.Board, depth: int = 3,
                        stockfish_path: str | None = None,
                        leaf_ms: int = 0) -> chess.Move:
    """Pick a move for side-to-move using minimax; optional leaf eval via local Stockfish."""
    engine = None
    try:
        if stockfish_path:
            import chess.engine
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"[warn] could not start Stockfish: {e}")

    color = 1 if board.turn == chess.WHITE else -1
    val, move = negamax(board, depth, -INF, INF, color, engine, leaf_ms)
    if engine:
        engine.quit()
    if move is None:
        # no legal moves
        legal = list(board.legal_moves)
        return legal[0] if legal else None
    return move

# ----- OPTIONAL: use a public Stockfish API instead of NCM -----
def best_move_from_cloud(board: chess.Board) -> str:
    """
    Calls chess-api.com to get the best move (UCI string).
    Returns e.g. 'e2e4'.
    """
    url = "https://chess-api.com/v1"
    r = requests.post(url, json={"fen": board.fen()})
    r.raise_for_status()
    data = r.json()
    return data.get("move")

if __name__ == "__main__":
    board = chess.Board()
    # Example: Agent vs Agent for a few moves
    for ply in range(12):
        move = choose_move_minimax(board, depth=3)  # bump to 4–5 if it's fast enough for you
        if move is None:
            break
        print(f"{ply+1}. {board.san(move)}  ({move.uci()})")
        board.push(move)
        if board.is_game_over():
            break
    print(board)
    print("Result:", board.result())
