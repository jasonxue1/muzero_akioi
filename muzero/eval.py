import akioi_2048 as ak
import printer
from muzero.choose_move import MoveChooser


def single_run(checkpoint_path):
    board = ak.init()
    # 1) 创建 chooser —— 只做一次
    chooser = MoveChooser(
        checkpoint_path=checkpoint_path,
        mode="mcts",  # "policy" or "mcts"
        mcts_simulations=10,  # 如果用 MCTS，就少量搜索更快
    )

    # 2) 在游戏循环里调用
    res = 0
    step = 0
    score = 0
    while not res:
        dir = chooser.choose(board)

        new_board, delta_score, res = ak.step(board, dir)

        if new_board != board:
            step += 1
        board = new_board
        score += delta_score
        # print(board)
    return (board, res, score, step)


def eval(num: int, checkpoint_path: str, log=False) -> list:
    score_list = []
    if not log:
        print(f"eval: 0/{num}", end="")
    for i in range(num):
        final_board, res, score, step = single_run(checkpoint_path)
        res = ("continue", "win", "lose")[res]
        if log:
            printer.print_table(final_board)
            print("\n")
            output = [["result", "score", "step"], [res, score, step]]
            printer.print_table(output)
            print("\n\n")
        else:
            print(f"\reval: {i + 1}/{num}   score: {score}", end="", flush=True)
        score_list.append(score)
    if not log:
        print("\neval: Done")

    return score_list


if __name__ == "__main__":
    checkpoint_path = "checkpoints/latest.pt"
    num = 3
    result = eval(num, checkpoint_path, True)
    result_table = [list(range(num)), result]
    printer.print_table(result_table)
