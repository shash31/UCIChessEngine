#include <chrono>
#include <thread>
#include "tt.h"
#include "Board.h"
#include "eval.h"

// TODOs:
// - Maybe implement killer move heuristic
// - Maybe implement history heuristic
// - Look into improving search (PVS, NegaScout, etc.)
// - Look into using NNUE as evaluation instead
// - Make into Lichess Bot account
// - Look into multithreaded search

constexpr int INF = 1e9;

uint64_t total_nodes;

TranspositionTable tt;

// Time limit for iterative deepening
std::chrono::time_point<std::chrono::steady_clock> start_time;
double soft_limit_ms;
double hard_limit_ms;

std::atomic<bool> stopSearch{false};
std::thread searchThread;

constexpr const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

inline void check_time() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    if (elapsed >= hard_limit_ms) {
        stopSearch = true;
    }
}

int quiescence(int ply, int alpha, int beta, Board& board) {
    total_nodes++;

    // Check time every 2048 nodes
    if ((total_nodes & 2047) == 0) check_time();

    // If time expired, abort search immediately
    if (stopSearch) return 0;

    int original_alpha = alpha;
    Move tt_move = 0;
    int tt_score = 0;

    if (tt.probe(board.hash, 0, alpha, beta, tt_score, tt_move, ply)) return tt_score;

    Square king_pos = static_cast<Square>(std::countr_zero(board.piece_type_bb[board.turn][KING]));
    bool in_check = board.is_square_attacked(king_pos);

    int stand_pat = -INF;
    if (!in_check) {
        stand_pat = pesto_eval(board);
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;
    }

    MoveList move_list;
    board.generate_pseudo_legal_moves(move_list, !in_check); // ALL moves if in check, ONLY captures if quiet
    board.score_moves(move_list, tt_move);

    int legal_moves_count = 0;
    Move best_move = 0;
    int best_score = in_check ? -INF : stand_pat;

    for (int i = 0; i < move_list.count; i++) {
        // Selection sort inline
        int best_index = i;
        for (int j = i + 1; j < move_list.count; j++) {
            if (move_list.scores[j] > move_list.scores[best_index]) best_index = j;
        }
        std::swap(move_list[i], move_list[best_index]);
        std::swap(move_list.scores[i], move_list.scores[best_index]);

        Move move = move_list[i];

        // Delta Pruning: Skip bad captures if not in check and can't reach alpha
        if (!in_check && ((move >> 12) < 8)) { // If not in check & not a promotion
            int captured_val = material_values[char_to_piece(board.grid[(move >> 6) & 0x3F])];
            if (stand_pat + captured_val + 200 < alpha) continue;
        }

        board.make_move(move);
        if (!board.is_king_capturable()) {
            legal_moves_count++;
            int score = -quiescence(ply + 1, -beta, -alpha, board);
            board.unmake_move(move);

            if (score > best_score) {
                best_score = score;
                best_move = move;

                if (score > alpha) alpha = score;
                if (alpha >= beta) break; // Cutoff
            }
        } else {
            board.unmake_move(move);
        }
    }

    if (in_check && legal_moves_count == 0) {
        return -INF + ply; // Actual Checkmate
    }

    if (!stopSearch) {
        TTFlag flag = (best_score <= original_alpha) ? UPPER : (best_score >= beta ? LOWER : EXACT);
        tt.store(board.hash, 0, best_score, flag, best_move, ply);
    }

    return best_score;
}

int negamax(int depth, int ply, Board& board, int alpha, int beta) {
    total_nodes++;

    // Check time every 2048 nodes
    if ((total_nodes & 2047) == 0) check_time();

    // If time expired, abort search immediately
    if (stopSearch) return 0;

    int original_alpha = alpha;
    Move tt_move = 0;
    int tt_score = 0;

    if (tt.probe(board.hash, depth, alpha, beta, tt_score, tt_move, ply) && (ply > 0)) return tt_score;

    if (depth == 0) return quiescence(ply + 1, alpha, beta, board);

    MoveList move_list;
    board.generate_pseudo_legal_moves(move_list);
    board.score_moves(move_list, tt_move);
    
    int best_score = -INF;
    Move best_move = 0;
    int legal_moves_count = 0;

    for (int i = 0; i < move_list.count; i++) {
        // MVV-LVA ordering
        int best_index = i;
        for (int j = i + 1; j < move_list.count; j++) {
            if (move_list.scores[j] > move_list.scores[best_index]) best_index = j;
        }

        std::swap(move_list[i], move_list[best_index]);
        std::swap(move_list.scores[i], move_list.scores[best_index]);

        Move move = move_list[i];
        board.make_move(move);
        if (!board.is_king_capturable()) {
            legal_moves_count++;
            int score = -negamax(depth - 1, ply + 1, board, -beta, -alpha);
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            alpha = std::max(best_score, alpha);
            if (alpha >= beta) {
                // std::println("cutoff reached"); 
                board.unmake_move(move);
                break;
            }
        }
        board.unmake_move(move);
    }

    if (legal_moves_count == 0) {
        Square king_pos = static_cast<Square>(std::countr_zero(board.piece_type_bb[board.turn][KING]));
        if (board.is_square_attacked(king_pos)) { // in check
            return -INF + ply; // Checkmate, return negative score with depth to prefer faster checkmates
        }
        return 0; // stalemate
    }

    if (!stopSearch) {
        TTFlag flag;
        if (best_score <= original_alpha) {
            flag = UPPER; // Fail-low
        } else if (best_score >= beta) {
            flag = LOWER; // Fail-high
        } else {
            flag = EXACT; // PV Node
        }

        tt.store(board.hash, depth, best_score, flag, best_move, ply);
    }

    return best_score;
}


Move ids(Board& board) {
    current_age++;

    // std::println("Current age: {}", current_age);

    // Iterative Deepening Search
    start_time = std::chrono::steady_clock::now();
    auto totalTime = 0;
    
    uint64_t all_nodes = 0;

    MoveList moves;
    board.generate_pseudo_legal_moves(moves, false);
    Move bestmove;
    // Setting bestmove to first legal move in case of time trouble as fallback
    for (int i = 0; i < moves.size(); i++) {
        board.make_move(moves[i]);
        if (!board.is_king_capturable()) {
            bestmove = moves[i];
            board.unmake_move(moves[i]);
            break;
        }
        board.unmake_move(moves[i]);
    }
    int alpha = -INF;
    int beta = INF;

    for (int depth = 1; depth < 30; depth++) { 
        auto start = std::chrono::steady_clock::now();
        total_nodes = 0;
        
        int score = negamax(depth, 0, board, alpha, beta);

        if (stopSearch) break; // Time ran out; Discard results

        // Retrieve best move found so far from TT for the root position
        Move tt_move = 0;
        int dummy_score = 0;
        tt.probe(board.hash, depth, alpha, beta, dummy_score, tt_move, 0);
        
        if (tt_move != 0) bestmove = tt_move;

        auto end = std::chrono::steady_clock::now();
        auto seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalTime += seconds;
        uint64_t nps = total_nodes*1000 / std::max<int64_t>(seconds, 1);
        all_nodes += total_nodes;

        // std::println("Score: {}", score);
        // std::println("Best move: {}", moveToString(bestmove));

        // std::cout << "Depth: " << depth 
        //         << " | Nodes: " << total_nodes 
        //         << " | Time: " << seconds << "s" 
        //         << " | NPS: " << static_cast<uint64_t>(nps) << std::endl;
        std::println("info depth {} score cp {} nodes {} nps {} time {} pv {}", depth, score, total_nodes, nps, totalTime, moveToString(bestmove)); std::fflush(stdout);

        // Stop early if a forced mate is found
        if (score > 90000 || score < -90000) break;

        if (totalTime > soft_limit_ms) break;
    }

    return bestmove;
}

void clearTablesAndHistory() {
    tt.clear();
    current_age = 0;
}

void initialize() {
    tt.resize(64); // 64 MB
    zobrist_keys.init();
    init_tables();
}

// Find the legal move matching a UCI string like "e2e4" or "e7e8q".
Move parseMove(Board& board, const std::string& str) {
    MoveList moves;
    board.generate_pseudo_legal_moves(moves, false);
    for (int i = 0; i < moves.size(); i++) { 
        if (moveToString(moves[i]) == str) return moves[i];
    }
    return Move{};  // "null" move: not found / illegal
}

void parsePosition(Board& board, std::istringstream& is) {
    std::string token;
    std::string fen;

    is >> token;
    if (token == "startpos") {
        fen = START_FEN;
        token.clear();
        is >> token;                       // "moves" or nothing
    } else if (token == "fen") {
        // A FEN is up to 6 space-separated fields; read until "moves" or end.
        while (is >> token && token != "moves") {
            if (!fen.empty()) fen += ' ';
            fen += token;
        }
    } else {
        return;                            // malformed, ignore
    }

    board.reset();
    board.parseFEN(fen);                   // std::string converts to string_view

    if (token == "moves") {
        while (is >> token) {
            Move m = parseMove(board, token);
            if (m == 0) break;       // adapt to your null-move check
            board.make_move(m);
        }
    }
}

void startSearch(Board& board) {
    stopSearch = false;
    searchThread = std::thread([&board] {
        Move best = ids(board); // prints info lines
        std::println("bestmove {}", moveToString(best)); std::fflush(stdout);
    });
}

// Parse arguments and set soft and hard time limits
void parseGo(Board& board, std::istringstream& is) {
    std::string token;
    int wtime;
    int btime;
    int winc;
    int binc;
    int movestogo = std::max(20, 50 - board.fullmove_no);
    bool moveTimeGiven = false;
    while (is >> token) {
        if (token == "wtime") {
            is >> token;
            wtime = std::stoi(token);
        }
        else if (token == "btime") {
            is >> token;
            btime = std::stoi(token);
        }
        else if (token == "winc") {
            is >> token;
            winc = std::stoi(token);
        }
        else if (token == "binc") {
            is >> token;
            binc = std::stoi(token);
        }
        else if (token == "movetime") {
            is >> token;
            soft_limit_ms = std::stoi(token);
            hard_limit_ms = std::stoi(token);
            moveTimeGiven = true;
        }
        else if (token == "movestogo") {
            is >> token;
            movestogo = std::stoi(token);
        }
        else if (token == "depth") {

        }
        else if (token == "nodes") {

        }
        else if (token == "infinite") {

        }
    }

    /* Time allocation formula
        overhead   = ~50-100          // network/process lag buffer
        movesLeft  = movestogo if given, else an estimate (e.g. 25-40)
        base       = (timeLeft - overhead) / movesLeft
        soft       = base + increment * ~0.75
        hard       = min(soft * 3-5, timeLeft * ~0.2)   // never close to your whole clock
    */
    if (!moveTimeGiven) {
        int timeLeft;
        int increment;
        if (board.turn == WHITE) {
            timeLeft = wtime; increment = winc;
        } else {
            timeLeft = btime; increment = binc;
        }
        int base = (timeLeft - 100) / movestogo;
        soft_limit_ms = base + increment * 0.75;
        hard_limit_ms = std::min(soft_limit_ms*4, timeLeft*0.2);
    }
    std::println(stderr, "Soft limit: {}; Hard limit: {}", soft_limit_ms, hard_limit_ms); std::fflush(stdout);

    startSearch(board);
}

void stopAndJoin() {
    stopSearch = true;
    if (searchThread.joinable()) searchThread.join();
}

void parseSetOption(std::istringstream& is) {

}

int main() {
    initialize();
    Board board;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd; is >> cmd;
        if (cmd == "uci") {
            std::cout << "id name MyC++Engine\n";
            std::cout << "id author Shashwat Dudeja\n";
            std::cout << "uciok" << std::endl;
        } 
        else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        }
        else if (cmd == "ucinewgame") {
            clearTablesAndHistory();
        }
        else if (cmd == "position") {
            stopAndJoin();
            parsePosition(board, is);
        } 
        else if (cmd == "go") {
            stopAndJoin();
            parseGo(board, is);
        } 
        else if (cmd == "setoption") {
            parseSetOption(is);
        }
        else if (cmd == "stop") {
            stopAndJoin();
        }
        else if (cmd == "quit") {
            stopAndJoin();
            break;
        }
    }
    return 0;
}

uint64_t perft(int depth, Board& board) { // For testing correctness of move generation
    if (depth == 0) return 1ULL;

    uint64_t nodes = 0;
    MoveList move_list;
    board.generate_pseudo_legal_moves(move_list);

    for (int i = 0; i < move_list.count; ++i) {
        Move move = move_list[i];

        board.make_move(move);
        if (board.is_king_capturable()) {
            board.unmake_move(move); 
            continue;
        }

        nodes += perft(depth - 1, board);

        board.unmake_move(move);
    }

    return nodes;
}

void perft_divide(int depth, Board& board, uint64_t& total_nodes) {
    MoveList move_list;
    board.generate_pseudo_legal_moves(move_list);

    for (int i = 0; i < move_list.count; ++i) {
        Move move = move_list[i];

        board.make_move(move);
        if (board.is_king_capturable()) {
            board.unmake_move(move);
            continue;
        }

        uint64_t nodes = perft(depth - 1, board);
        total_nodes += nodes;
        board.unmake_move(move);

        uint8_t to = (move >> 6) & 0x3F;
        uint8_t from = move & 0x3F;
        std::println("{}{}: {}", getStringFromSquare(from), getStringFromSquare(to), nodes);
    }

    std::cout << "\nTotal Nodes at Depth " << depth << ": " << total_nodes << "\n";
}
