import numpy as np
import random

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
# 問題２で定義した outer_prod を定義してください


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

# 予測値を求める関数 neural_network の定義
# 問題２で定義した neural_network を使用してください


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
# 問題２で定義した grad_descent_learn を使用してください


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


# 重みを生成する関数 create_weights の定義
'''
関数名：create_weights
引数：
  input_layer_number = 入力層のユニット数
  output_layer_number = 出力層のユニット数
処理：
  行の数 = 出力層のユニット数、列の数 = 入力層のユニット数の行列を生成する
  ただし行列の要素には０以上１以下の実数を配置する
戻り値
  生成した重みの行列
'''


def create_weights(input_layer_number, output_layer_number):
    # 出力層のユニット数　ｘ　入力層のユニット数 の０行列（要素がすべて０の行列）を生成する
    out_matrix = np.zeros((input_layer_number, output_layer_number))
    # 行列の要素に０以上１位かの実数を設定する（２重ループ）
    # 出力層のユニット数分繰り返す
    for i in range(output_layer_number):
        # 入力層のユニット数分繰り返す
        for j in range(input_layer_number):
            # 行列の i 行、j 列に０以上１以下の乱数実数を保存する
            out_matrix[i][j] = random.uniform(0, 1)
    # 行列を返す
    return out_matrix


# 図形文字データの初期化
# まる　タイプ１
fig_maru_1_str = [
    "          ",
    "   ****   ",
    "  *    *  ",
    " *      * ",
    " *      * ",
    " *      * ",
    " *      * ",
    "  *    *  ",
    "   ****   ",
    "          "
]

# まる　タイプ２
fig_maru_2_str = [
    "          ",
    "  ****    ",
    " *    *   ",
    "*      *  ",
    "*      *  ",
    "*      *  ",
    "*      *  ",
    " *    *   ",
    "  ****    ",
    "          "
]

# まる　タイプ３
fig_maru_3_str = [
    "          ",
    "    ****  ",
    "   *    * ",
    "  *      *",
    "  *      *",
    "  *      *",
    "  *      *",
    "   *    * ",
    "    ****  ",
    "          "
]

# まる　タイプ４
fig_maru_4_str = [
    "          ",
    "   *****  ",
    "  *     * ",
    " *       *",
    " *       *",
    " *       *",
    " *       *",
    "  *     * ",
    "   *****  ",
    "          "
]

#

# ばつ　タイプ１
fig_batsu_1_str = [
    "          ",
    " **      *",
    "  **    * ",
    "   **  *  ",
    "    **    ",
    "   ** *   ",
    "  **   *  ",
    " **     * ",
    "          ",
    "          "
]

# ばつ　タイプ２
fig_batsu_2_str = [
    "          ",
    " *      **",
    "  *    ** ",
    "   *  **  ",
    "    ***   ",
    "   *  **  ",
    "  *    ** ",
    " *      **",
    "          ",
    "          "
]

# ばつ　タイプ３
fig_batsu_3_str = [
    "          ",
    " **     * ",
    "  **   *  ",
    "   ** *   ",
    "    **    ",
    "   * **   ",
    "  *   **  ",
    " *     ** ",
    "          ",
    "          "
]

# ばつ　タイプ４
fig_batsu_4_str = [
    "          ",
    "  *      *",
    "   *    * ",
    "    *  *  ",
    "     **   ",
    "    *  *  ",
    "   *    * ",
    "  *      *",
    "          ",
    "          "
]

# 認識させる まる
new_maru_str = [
    "          ",
    "   ****   ",
    "  **  **  ",
    " **    ** ",
    " **    ** ",
    " **    ** ",
    " **    ** ",
    "  **  **  ",
    "   ****   ",
    "          "
]
# 認識させる ばつ
new_batsu_str = [
    "          ",
    " **    ** ",
    "  **  **  ",
    "   ****   ",
    "    **    ",
    "   ****   ",
    "  **  **  ",
    " **    ** ",
    "          ",
    "          "
]

# 文字列配列を数値配列に変換する関数 str_to_number_list の定義
'''
関数名：str_to_number_list
引数：str = 図形（まる、ばつ）の文字列のリスト
処理：図形の文字列のリストを数値リストに変換
戻り値：数値リスト
'''


def str_to_number_list(str):
    # 数値リストの初期化
    number_patten = []
    # 文字列のリストを数値リストに変換（２重ループ）
    # 文字 '*' なら数値の 1 に、それ以外なら数値の 0 に変換して数値リストに追加
    # 行数分繰り返す
    for i in range(len(str)):
        # 列数分（文字数分）繰り返す
        for j in range(len(str[i])):
            # 文字が '*' なら
            if str[i][j] == '*':
                # 数値リストに 1 を追加する
                number_patten.append(1)
            # そうでなければ
            else:
                # 数値リストに 0 を追加する
                number_patten.append(0)
    # 数値リストを返す
    return number_pattern


# 学習処理関数 learn_task の定義
'''
関数名：learn_task(input_layer, truth, weights, alpha, number)
引数：
  input_layer = 入力層リスト
  truth = 目的値リスト
  weights = 重み行列
  alpha = 学習効率
  number = 学習させる回数
処理：勾配降下法を使用して指定された回数学習する
戻り値：なし
'''


def learn_task(input_layer, truth, weights, alpha, number):
    # number 回、学習する
    for iter in range(number):
        # タイプ１からタイプ４の「まる」の学習

        # 予測値リストを求める
        pred = neural_network(input_layer, weights)
        # 学習する（重みを調整して更新する）
        weights = grad_descent_learn(input_layer, truth, pred, weights, alpha)


# 「まる」か「ばつ」を予測する関数 figure_recognition の定義
'''
関数名：figure_recognition
引数：pred = 予測値リスト
処理：
  予測値リストの最大値のインデックス番号（リストの要素番号を求めて）
  '〇'、'✕'、'その他' を返す
戻り値：'〇' または '✕' または 'その他'
'''


def figure_recognition(pred):
    # max メソッド, index メソッドを使用して、予測値リスト pred の最大値のインデックス番号を求める
    max_index = pred.index(max(pred))
    # 図形を判断する
    # 「まる」かどうかの判断

    # '〇' を返す
    return '〇'
    # 「ばつ」かどうかの判断

    # '✕' を返す
    return '✕'
    # その他のとき

    # 'その他' を返す
    return 'その他'


# 入力データの設定
# まる　タイプ１
maru_1 = fig_maru_str_list(fig_maru_1_str)
# まる　タイプ２
maru_2 = fig_maru_str_list(fig_maru_2_str)
# まる　タイプ３
maru_3 = fig_maru_str_list(fig_maru_3_str)
# まる　タイプ４
maru_4 = fig_maru_str_list(fig_maru_4_str)
# 「まる」の入力データの設定
input_maru = [maru_1, maru_2, maru_3, maru_4]

# ばつ　タイプ１
batsu_1 = fig_batsu_str_list(fig_batsu_1_str)
# ばつ　タイプ２
batsu_2 = fig_batsu_str_list(fig_batsu_2_str)
# ばつ　タイプ３
batsu_3 = fig_batsu_str_list(fig_batsu_3_str)
# ばつ　タイプ４
batsu_4 = fig_batsu_str_list(fig_batsu_4_str)
# 「ばつ」の入力データの設定
input_batsu = [batsu_1, batsu_2, batsu_3, batsu_4]

# 目的値（正解）の設定
truth = [
    # まる
    [1, 0, 0],
    # ばつ
    [0, 1, 0],
    # その他
    [0, 0, 1]
]

# 重みを生成
weights = [0.1, 0.1, 0.1]

# アルファの初期化
alpha = 0.001

print('「まる」の学習開始')
# 「まる」を５００回学習する
for i in range(len(input_maru)):
    learn_task(input_maru[i], truth, weights, alpha, 500)

print('学習終了')

print('「ばつ」の学習開始')
# 「ばつ」を５００回学習する
for i in range(len(input_batsu)):
    learn_task(input_batsu[i], truth, weights, alpha, 500)

print('学習終了')

# 予測させる「まる」の入力データを設定
input_new_maru = new_maru_str
# 予測させる「ばつ」の入力データを設定
input_new_batsu = new_batsu_str

# 新しいタイプの「まる」を予測
pred = neural_network(input_new_maru, weights)
# 予測した図形を表示
print('予測した図形 = {}'.format(figure_recognition(pred)))
# 新しいタイプの「ばつ」を予測
pred = neural_network(input_new_batsu, weights)
# 予測した図形を表示
print('予測した図形 = {}'.format(figure_recognition(pred)))
