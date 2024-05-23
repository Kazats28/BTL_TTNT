from color_recognition_api import knn_classifier
import sys
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def process_image(i):
    prediction = knn_classifier.main('training.csv', 'test.csv', i)
    return prediction


def main(folder_path):
    total_img = 0
    detect_true = 0
    y_true = []
    y_pred = []
    red = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    white = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    green = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    blue = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    black = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    violet = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    yellow = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    grey = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    brown = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    orange = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    tp = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    fp = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    fn = {'blue': 0, 'green': 0, 'black': 0, 'brown': 0, 'red': 0, 'white': 0, 'orange': 0, 'violet': 0, 'yellow': 0,
           'grey': 0}
    with open(f'{folder_path}.csv', 'r') as input_file:
        csv_reader = csv.reader(input_file)
        for row in csv_reader:
            total_img += 1
            feature_data = row[0] + ',' + row[1] + ',' + row[2]
            with open('test.csv', 'w') as output_file:
                output_file.write(feature_data)
            prediction = process_image(8)
            y_true.append(f'{row[3]}')
            y_pred.append(prediction)
            if prediction == f'{row[3]}':
                detect_true += 1
                tp[prediction] += 1
            else:
                fn[f'{row[3]}'] += 1
                fp[prediction] += 1
            if f'{row[3]}' == 'red':
                if f'{row[3]}' == prediction:
                    red[f'{row[3]}'] += 1
                else:
                    red[prediction] += 1
            if f'{row[3]}' == 'white':
                if f'{row[3]}' == prediction:
                    white[f'{row[3]}'] += 1
                else:
                    white[prediction] += 1
            if f'{row[3]}' == 'green':
                if f'{row[3]}' == prediction:
                    green[f'{row[3]}'] += 1
                else:
                    green[prediction] += 1
            if f'{row[3]}' == 'blue':
                if f'{row[3]}' == prediction:
                    blue[f'{row[3]}'] += 1
                else:
                    blue[prediction] += 1
            if f'{row[3]}' == 'black':
                if f'{row[3]}' == prediction:
                    black[f'{row[3]}'] += 1
                else:
                    black[prediction] += 1
            if f'{row[3]}' == 'violet':
                if f'{row[3]}' == prediction:
                    violet[f'{row[3]}'] += 1
                else:
                    violet[prediction] += 1
            if f'{row[3]}' == 'yellow':
                if f'{row[3]}' == prediction:
                    yellow[f'{row[3]}'] += 1
                else:
                    yellow[prediction] += 1
            if f'{row[3]}' == 'grey':
                if f'{row[3]}' == prediction:
                    grey[f'{row[3]}'] += 1
                else:
                    grey[prediction] += 1
            if f'{row[3]}' == 'brown':
                if f'{row[3]}' == prediction:
                    brown[f'{row[3]}'] += 1
                else:
                    brown[prediction] += 1
            if f'{row[3]}' == 'orange':
                if f'{row[3]}' == prediction:
                    orange[f'{row[3]}'] += 1
                else:
                    orange[prediction] += 1

    print(f'dectedTrue: {(detect_true/total_img)*100}%')
    print("__________________________________________________________")
    print(f'blue: {blue}')
    print(f'green: {green}')
    print(f'black: {black}')
    print(f'brown: {brown}')
    print(f'red: {red}')
    print(f'white: {white}')
    print(f'orange: {orange}')
    print(f'violet: {violet}')
    print(f'yellow: {yellow}')
    print(f'grey: {grey}')
    print(f'TP: {tp}')
    print(f'FN: {fn}')
    print(f'FP: {fp}')
    target_names = ['blue', 'green', 'black', 'brown', 'red', 'white', 'orange', 'violet', 'yellow', 'grey']
    print(classification_report(y_true, y_pred, labels=target_names))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, xticks_rotation=45, cmap='Blues')
    plt.show()
    
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "train"
    main(folder_path)
    