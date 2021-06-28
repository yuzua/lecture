# 入力層から隠れ層への重みの初期化
ih_wgt = [
    # 中間層・１個目のユニットへの重み
    [0.78, 0.64, 0.55, 0.45],
    # 中間層・２個目のユニットへの重み
    [0.79, 0.57, 0.83, 0.73]
]

# 隠れ層から出力層への重みの初期化
hp_wgt = [
    # 出力層・売上げユニットへの重み
    [0.88, 0.73],
    # 出力層・来客数ユニットへの重み
    [0.68, 0.67],
    # 出力層・リピート数ユニットへの重み
    [0.75, 0.59]
]

# 重みをまとめる
# [入力層と隠れ層間の重み, 隠れ層と出力層間の重み]
weights = [ih_wgt, hp_wgt]

# ベクトルの加重和を求める関数 w_sum の定義
'''
関数名：w_sum(a, b)
引数：a, b = 数値のリスト
処理：２つの数値リスト a と b の加重和を求める
戻り値：加重和
'''


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


def vect_mat_mul(vect, matrix):
    # 加重和のリストを初期化する
    output = []
    # 行列の行数分繰り返す
    for i in range(len(matrix)):
        # w_sum 関数を使用して加重和を求め、加重和のリストに追加する
        output.append(w_sum(vect, matrix[i]))
    # 加重和のリストを返す
    return output


# 予測値を求める関数 neural_network(input, weights) の定義
'''
関数名：neural_network
引数：input = 入力リスト, weights = 重み行列
処理：予測値を求める
戻り値：出力層の予測値リスト
'''


def neural_network(input, weights):
    # 隠れ層（中間層）の予測値リストを求める
    hid = vect_mat_mul(input, weights[0])
    # 出力層の予測値リストを求める
    pred = vect_mat_mul(hid, weights[1])
    # 出力層の予測値のリストを返す
    return pred


# 入力層の初期化
input = [0.25, 0.78, 0.66, 0.35]
# 予測する
pred = neural_network(input, weights)
# 予測結果を表示する
print('売上げ = {0:.3f}\n来客数 = {1:.3f}\nリピート数 = {2:.3f}'
      .format(pred[0], pred[1], pred[2]))
