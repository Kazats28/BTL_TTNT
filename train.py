from color_recognition_api import knn_classifier
import sys
import matplotlib.pyplot as plt
import csv


def process_image(color, detect_true, i):
    prediction = knn_classifier.main('training.csv', 'test.csv', i)
    if prediction == color:
        detect_true = detect_true + 1
    return detect_true


def main(folder_path):
    for j in range(1, 6):
        x = []
        y = []

        open('training.csv', 'w').close()
        for k in range(1, 6):
            if k == j:
                continue
            with open(f'{folder_path}{k}.csv', 'r') as input_file:
                with open('training.csv', 'a', newline='') as output_file:
                    csv_reader = csv.reader(input_file)
                    csv_writer = csv.writer(output_file)
                    for row in csv_reader:
                        csv_writer.writerow(row)

        for i in range(1, 31):
            total_img = 300
            detect_true = 0
            x.append(i)
            with open(f'{folder_path}{j}.csv', 'r') as input_file:
                csv_reader = csv.reader(input_file)
                for row in csv_reader:
                    feature_data = row[0] + ',' + row[1] + ',' + row[2]
                    with open('test.csv', 'w') as output_file:
                        output_file.write(feature_data)
                    detect_true = process_image(f'{row[3]}', detect_true, i)

            y.append((detect_true/total_img)*100)
            print(f'dectedTrue: {detect_true}, totalImg: {total_img}, k: {i}')
            print("__________________________________________________________")

        plt.plot(x, y, marker='o', label=f'Test {j}')

    plt.title('Biểu đồ')
    plt.xlabel('K')
    plt.ylabel('Độ chính xác')
    plt.xticks(range(1, 31))

    # Thêm lưới
    plt.grid(True)

    plt.legend()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "train"
    main(folder_path)
    