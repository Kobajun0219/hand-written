"""
    SVM適用前のデータの前処理を行います.
    MNISTファイル(gzip)を、CSVファイルに変換します.
"""
import os
import struct
import gzip


if __name__ == "__main__":

    if not os.path.exists("csv"):
        os.mkdir("csv")

    """
        **** ここを実装します（基礎課題） ****
        `mnist`フォルダにあるデータから、CSVを作成し、`csv`フォルダに出力するプログラムを作成してください。
        実装方法は、講義資料や答えを参照してください。
        最初の課題から難易度高めですが、ぜひチャレンジしてみてください！

        作成が完了したら、同ディレクトリにある`check_image.py`を実行し、
        画像が正しく出力されるかを確認してください。
    """
    # # Read MNIST `label`.
    # fpath = "./mnist/train-labels-idx1-ubyte.gz"
    # with gzip.open(fpath, "rb") as f:
    #     magic_number, img_count = struct.unpack(">II", f.read(8))
    #     labels = []
    #     for i in range(img_count):
    #         label = str(struct.unpack("B", f.read(1))[0])
    #         labels.append(label)

    # # Write as csv.
    # outpath = './csv/training-labels.csv'
    # with open(outpath, "w") as f:
    #     f.write("\n".join(labels))

    # # Read MNIST `images`.
    # fpath = "./mnist/train-images-idx3-ubyte.gz"
    # with gzip.open(fpath, "rb") as f:
    #     _, img_count = struct.unpack(">II", f.read(8))
    #     rows, cols = struct.unpack(">II", f.read(8))
    #     images = []
    #     for i in range(img_count):
    #         binary = f.read(rows * cols)
    #         images.append(",".join([str(b) for b in binary]))

    # # Write as csv.
    # outpath = './csv/training_image.csv'
    # with open(outpath, "w") as f:
    #     f.write("\n".join(images))

        # Read MNIST `label`.
    fpath = "./mnist/t10k-labels-idx1-ubyte.gz"
    with gzip.open(fpath, "rb") as f:
        magic_number, img_count = struct.unpack(">II", f.read(8))
        labels = []
        for i in range(img_count):
            label = str(struct.unpack("B", f.read(1))[0])
            labels.append(label)

    # Write as csv.
    outpath = './csv/test-labels.csv'
    with open(outpath, "w") as f:
        f.write("\n".join(labels))

    # Read MNIST `images`.
    fpath = "./mnist/t10k-images-idx3-ubyte.gz"
    with gzip.open(fpath, "rb") as f:
        _, img_count = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        images = []
        for i in range(img_count):
            binary = f.read(rows * cols)
            images.append(",".join([str(b) for b in binary]))

    # Write as csv.
    outpath = './csv/test_images.csv'
    with open(outpath, "w") as f:
        f.write("\n".join(images))
