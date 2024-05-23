import cv2
import threading
from tkinter import Tk, Button, filedialog
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier


def select_image():
    # Hiển thị hộp thoại chọn file để chọn ảnh
    filepath = filedialog.askopenfilename()
    if filepath:
        # Tạo một luồng riêng cho việc xử lý và hiển thị ảnh với OpenCV
        threading.Thread(target=process_image, args=(filepath,)).start()


def process_image(filepath):
    # Đọc ảnh được chọn
    source_image = cv2.imread(filepath)

    # Kiểm tra và hiển thị ảnh
    if source_image is not None:
        # Thực hiện nhận diện màu
        prediction = detect_color(source_image)
        cv2.destroyAllWindows()
        # Tạo cửa sổ OpenCV
        window_name = f'Color Image: {prediction}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Tính toán kích thước font dựa trên độ phân giải của ảnh
        img_height, img_width = source_image.shape[:2]
        font_scale = img_width / 200  # Điều chỉnh giá trị này để phù hợp với nhu cầu của bạn
        font = cv2.FONT_HERSHEY_PLAIN
        thickness = int(font_scale * 2)  # Điều chỉnh độ dày của văn bản dựa trên font_scale
        # Tính toán kích thước văn bản
        text = 'Color: ' + prediction
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Tính toán vị trí để văn bản nằm giữa cửa sổ
        text_x = (img_width - text_size[0]) // 2
        text_y = (img_height + text_size[1]) // 2

        # Hiển thị kết quả trên ảnh
        cv2.putText(
            source_image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 255),
            thickness
        )

        cv2.imshow(f'Color Image: {prediction}', source_image)

        # Vòng lặp kiểm tra sự kiện để đảm bảo Tkinter tiếp tục hoạt động
        while cv2.getWindowProperty(f'Color Image: {prediction}', cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(50) & 0xFF == 27:  # Thoát nếu nhấn phím Esc
                break

    else:
        print("Failed to read the selected image.")


def detect_color(image):
    # Nhận diện màu
    color_histogram_feature_extraction.color_histogram_of_test_image(image)
    prediction = knn_classifier.main('training.csv', 'test.csv', 8)
    print('Detected color is:', prediction)
    return prediction


if __name__ == "__main__":
    # Gọi hàm chọn ảnh khi chương trình được chạy
    root = Tk()
    root.title("Nhận diện màu")
    root.geometry("300x300")

    select_another_button = Button(root, text="Chọn ảnh", command=select_image)
    select_another_button.pack(pady=100)

    root.mainloop()
