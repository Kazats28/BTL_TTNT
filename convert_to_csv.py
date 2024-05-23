from color_recognition_api import color_histogram_feature_extraction

if __name__ == "__main__":
    for i in range(1, 6):
        color_histogram_feature_extraction.training(i)
