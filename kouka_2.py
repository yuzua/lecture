import numpy as np

# ベクトルの加重和を求める関数 w_sum の定義
'''
関数名：w_sum(a, b)
引数：a, b = 数値のリスト
処理：２つの数値リスト a と b の加重和を求める
戻り値：加重和
'''
# 問題１で定義した w_sum を使用してください


def w_sum(a, b):
    # ２つの数値リストの長さが等しいときだけ以下を実行する
    assert(len(a) == len(b))
    # 加重和を０に初期化
    output = 0
    # 数値リストの長さ分繰り返す
    for i in range(len(a)):
        # ２つの数値リストの加重和を求める
        output += (a[i] * b[i])
    # 加重和を返す
    return output


# 数値リストと行列の加重和を求める関数 vect_mat_mul(vect, matrix) の定義
'''
関数名：vect_mat_mul
引数：vect = 数値リスト、matrix = 行列
処理：数値リストと行列の加重和を求める
戻り値：加重和のリスト
'''
# 問題１で定義した vect_mat_mul を使用してください


def vect_mat_mul(vect, matrix):
    # 加重和のリストを初期化する
    output = []
    # 行列の行数分繰り返す
    for i in range(len(matrix)):
        # w_sum 関数を使用して加重和を求め、加重和のリストに追加する
        output.append(w_sum(vect, matrix[i]))
    # 加重和のリストを返す
    return output


# ２つのリストの要素間を総なめして要素ごとのかけ算をして
# その結果を行列に格納する関数 outer_prod の定義
'''
関数名：outer_prod(a, b)
引数：a = 数値のリスト、b = 数値のリスト
処理：リスト a と b の要素を総なめにしてかけ算をし行列に配置する
  ⇒ outer_prod([1, 2, 3], [4, 5, 6])
          ↓
      [
        [1 * 4, 1 * 5, 1 * 6],
        [2 * 4, 2 * 5, 2 * 6],
        [3 * 4, 3 * 5, 3 * 6]
      ]
戻り値：リスト a と b の要素を総なめにしてかけ算した結果を保存した行列
'''


def outer_prod(a, b):
    # a 行 b 列のゼロ行列を生成する
    out = np.zeros((len(a), len(b)))
    # a 行 b 列の行列にリスト a とリスト b の
    # 要素を総なめしたかけ算を行い、行列に格納する
    # リスト a の長さ分繰り返す
    for i in range(len(a)):
        # リスト b の長さ分繰り返す
        for j in range(len(b)):
            # リスト a の　i 番目の要素と、リスト b の j 番目の要素を掛けて、
            # 行列の i 行 j 列目に保存する
            out[i][j] = a[i] * b[j]
    # 要素間のかけ算の結果を保存した行列を返す
    return out


# 予測値を求める関数 neural_network(input, weights) の定義
'''
関数名：neural_network
引数：input = 入力リスト, weights = 重み行列
処理：予測値を求める
戻り値：出力層の予測値リスト
'''


def neural_network(input, weights):
    # 入力値リストと重み行列の加重和を求める
    pred = vect_mat_mul(input, weights)
    # 予測値のリストを返す
    return pred


# 学習関数 grad_descent_learn(input, truth, pred, weights, alpha) の定義
'''
関数名：grad_descent_learn
引数：
    input：入力値リスト
    truth：目的値リスト
    pred：予測値リスト
    weights：重み行列
    alpha：学習効率値
処理：勾配降下法に基づき重みを修正する
戻り値：修正された重み行列
'''


def grad_descent_learn(input, truth, pred, weights, alpha):
    # デルタリストの初期化
    delta = []
    # 予測値の数分繰り返す
    for i in range(len(truth)):
        # デルタを求めてリストに追加する
        delta.append(pred[i] - truth[i])
    # 重みの微調整量を求め行列に格納する
    weight_deltas = outer_prod(delta, input)
    # 重み行列を更新する
    # 重み行列の行数分繰り返す
    for i in range(len(weights)):
        # 重み行列の列数分繰り返す
        for j in range(len(weights[i])):
            # アルファ（学習効率値）と重みの微調整量を使用して重み行列を修正する
            weights[i][j] -= alpha * weight_deltas[i][j]
    # 重み行列を返す
    return weights


# 重みの初期化
weights = [
    # 売上げへの重み
    [0.78, 0.64, 0.55, 0.45],
    # 来客数への重み
    [0.79, 0.57, 0.83, 0.73]
]

# 入力層の初期化
input = [0.25, 0.78, 0.66, 0.35]

# print(neural_network(input, weights))
# 目的値の初期化
truth = [1.0, 1.5]

# アルファ（学習効率値）の初期化
alpha = 0.1

# 勾配降下法による学習（５０回学習）
for i in range(50):
    # 予測値を求める
    pred = neural_network(input, weights)
    # 学習する
    weights = grad_descent_learn(input, truth, pred, weights, alpha)
    # 初速値を表示
    print('予測値 = [売上げ = {:.8f}、来客数 = {:.8f}]'.format(pred[0], pred[1]))
