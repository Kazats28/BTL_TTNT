import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier


def main():
    # Khởi tạo camera
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    # Kiểm tra xem camera có mở thành công không
    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        return

    while True:
        # Đọc khung hình từ camera
        ret, frame = cap.read()

        # Kiểm tra xem khung hình có được đọc thành công không
        if not ret:
            print("Failed to grab frame")
            break

        # Nhận diện màu sắc
        color_histogram_feature_extraction.color_histogram_of_test_image(frame)
        prediction = knn_classifier.main('training.csv', 'test.csv', 5)
        cv2.putText(frame, 'Prediction: ' + prediction, (15, 45), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        # Hiển thị khung hình
        cv2.imshow('color classifier', frame)

        # Thoát khỏi vòng lặp nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
