"""
    SVMアルゴリズムで手書き文字の判定を学習し、また結果を評価します.
"""
import os
from sklearn import svm
from sklearn import metrics
import joblib

if __name__ == "__main__":

    if not os.path.exists("result"):
        os.mkdir("result")

    """
        **** ここを実装します（基礎課題） ****
        `csv`フォルダからデータを読み込み、SVMアルゴリズムを用いた学習を行ってください。
        そして学習結果を`result`フォルダに`svm.pkl`という名前で保存してください。

        実装ステップ：
            ・トレーニングデータを読み込む
            ・SVGアルゴリズムによる学習を行う
            ・テストデータを読み込む
            ・精度とメトリクスによる性能評価を行う
            ・学習結果を`result/svm.pkl`ファイルとして保存する

        参考になる情報
            講義スライドや答えを適宜確認しながら実装してみてください。
            サンプルを見ながら手を動かしながら学ぶという感じがお勧めです。

        ここが一番大変なところです。
        ぜひぜひ頑張ってください！！
    """

    with open("./csv/training_image.csv") as f:
        images = f.read().split("\n")[:20000]
    with open("./csv/training-labels.csv") as f:
        labels = f.read().split("\n")[:20000]

    #Convert data.
    images = [[int(i)/256 for i in image.split(",")] for image in images]
    labels = [int(l) for l in labels]

    #Use SVM.
    clf = svm.SVC()
    clf.fit(images, labels)

    # テストデータを用いて学習した結果から予測を行い、精度を評価します
    with open("./csv/test_images.csv") as f:
        test_images = f.read().split("\n")[:500]
    with open("./csv/test-labels.csv") as f:
        test_labels = f.read().split("\n")[:500]

    #Convert data.
    test_images = [[int(i)/256 for i in image.split(",")] for image in test_images]
    test_labels = [int(l) for l in test_labels]

    #Predict.
    predict = clf.predict(test_images)

    #Show results.
    ac_score = metrics.accuracy_score(test_labels, predict)
    print("Accracy:", ac_score)
    cl_report = metrics.classification_report(test_labels, predict)
    print(cl_report)

    joblib.dump(clf, "./result/svm.pkl")
