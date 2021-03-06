from sklearn import svm

tdata = [
    [19, 25, 230, 110, 206, 28, 105, 320, 136, 0, 223, 318, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0],
    [88, 317, 56, 0, 11, 144, 172, 31, 0, 28, 173, 317, 158, 17, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 51, 56, 317, 184, 24, 105, 0, 20, 32, 165, 322, 0, 78, 0, 0, 0, 0, 0, 0, 0, 0],
    [126, 0, 261, 296, 7, 134, 269, 174, 53, 20, 179, 42, 0, 62, 0, 0, 0, 0, 0, 0, 0, 0],
    [77, 7, 261, 288, 0, 92, 263, 189, 60, 148, 115, 0, 18, 37, 0, 0, 0, 0, 0, 0, 0, 0],
    [243, 176, 50, 0, 0, 127, 239, 293, 175, 73, 10, 42, 90, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [188, 26, 234, 165, 0, 252, 55, 78, 36, 328, 124, 0, 226, 90, 0, 0, 0, 0, 0, 0, 0, 0],
    [257, 90, 0, 281, 166, 0, 224, 22, 119, 48, 76, 294, 254, 121, 0, 0, 0, 0, 0, 0, 0, 0],
    [245, 88, 96, 9, 0, 313, 205, 27, 0, 189, 121, 0, 53, 313, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 76, 139, 34, 150, 185, 36, 0, 88, 0, 1, 25, 66, 188, 0, 0, 0, 0, 0, 0, 0, 0],
    [104, 0, 170, 354, 0, 36, 141, 30, 64, 356, 150, 73, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [20, 5, 135, 26, 48, 149, 0, 64, 126, 148, 92, 0, 141, 76, 0, 0, 0, 0, 0, 0, 0, 0],
    [138, 334, 16, 15, 133, 20, 47, 337, 0, 72, 147, 77, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [57, 0, 170, 349, 141, 38, 68, 353, 3, 23, 0, 68, 107, 6, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 73, 139, 35, 35, 0, 146, 153, 32, 130, 91, 5, 3, 21, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 18, 103, 13, 38, 135, 30, 0, 119, 153, 18, 101, 52, 154, 0, 0, 0, 0, 0, 0, 0, 0],
    [77, 340, 25, 0, 98, 8, 175, 337, 14, 102, 0, 26, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [108, 11, 113, 350, 3, 19, 20, 353, 54, 0, 120, 25, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0],
    [37, 0, 49, 163, 106, 17, 5, 92, 110, 163, 0, 13, 56, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 41, 114, 110, 121, 14, 34, 172, 76, 0, 92, 173, 10, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 33, 104, 347, 112, 5, 7, 349, 62, 0, 115, 95, 0, 51, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 24, 140, 356, 101, 12, 37, 361, 0, 44, 58, 0, 114, 35, 0, 0, 0, 0, 0, 0, 0, 0],
    [96, 15, 19, 172, 2, 25, 0, 73, 114, 40, 92, 176, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 65, 153, 45, 95, 0, 165, 147, 39, 133, 40, 18, 6, 113, 0, 0, 0, 0, 0, 0, 0, 0],
    [175, 189, 77, 0, 0, 64, 171, 69, 0, 114, 133, 19, 31, 139, 0, 0, 0, 0, 0, 0, 0, 0],
    [26, 30, 304, 304, 10, 128, 304, 172, 133, 16, 0, 77, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 125, 293, 99, 75, 0, 293, 184, 0, 75, 45, 144, 35, 27, 0, 0, 0, 0, 0, 0, 0, 0],
    [152, 14, 178, 128, 60, 36, 86, 187, 173, 55, 0, 180, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [164, 80, 0, 133, 101, 0, 45, 43, 2, 249, 158, 132, 143, 25, 0, 0, 0, 0, 0, 0, 0, 0],
    [286, 77, 5, 341, 262, 16, 0, 216, 178, 25, 224, 0, 274, 124, 0, 0, 0, 0, 0, 0, 0, 0],
    [294, 122, 0, 171, 216, 12, 290, 20, 143, 188, 306, 65, 261, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 23, 5, 359, 101, 291, 89, 46, 38, 0, 98, 367, 77, 13, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 8, 30, 254, 97, 23, 109, 250, 105, 64, 57, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 0],
    [111, 72, 4, 9, 27, 111, 100, 22, 0, 62, 108, 126, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 70, 99, 19, 97, 122, 2, 7, 27, 116, 110, 67, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 15, 55, 250, 101, 17, 116, 244, 56, 0, 0, 64, 116, 71, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 21, 15, 366, 98, 20, 103, 360, 109, 75, 59, 0, 114, 306, 0, 0, 0, 0, 0, 0, 0, 0],
    [12, 10, 107, 356, 105, 29, 10, 360, 61, 0, 113, 80, 0, 55, 0, 0, 0, 0, 0, 0, 0, 0],
    [116, 83, 67, 0, 28, 254, 108, 25, 10, 7, 106, 254, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0],
    [68, 0, 20, 123, 10, 4, 117, 82, 109, 144, 0, 62, 110, 28, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 27, 40, 112, 86, 10, 98, 112, 16, 84, 2, 8, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [44, 228, 2, 8, 85, 8, 110, 226, 9, 74, 31, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0],
    [74, 2, 114, 364, 0, 15, 14, 368, 33, 0, 115, 331, 93, 17, 0, 0, 0, 0, 0, 0, 0, 0],
    [112, 22, 98, 360, 19, 26, 0, 360, 62, 0, 106, 11, 12, 36, 0, 0, 0, 0, 0, 0, 0, 0],
    [108, 15, 88, 239, 7, 27, 52, 0, 21, 239, 0, 42, 101, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [54, 0, 93, 97, 3, 26, 102, 10, 28, 128, 0, 46, 82, 127, 0, 0, 0, 0, 0, 0, 0, 0],
    [201, 120, 23, 11, 27, 88, 58, 0, 0, 51, 132, 129, 96, 16, 0, 0, 0, 0, 0, 0, 0, 0],
    [96, 10, 206, 154, 0, 75, 10, 23, 31, 104, 208, 86, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [231, 53, 0, 144, 99, 160, 187, 0, 219, 86, 162, 8, 213, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [148, 0, 1, 187, 179, 57, 0, 102, 165, 87, 115, 6, 167, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [51, 344, 12, 0, 169, 159, 0, 224, 160, 344, 173, 237, 18, 277, 0, 0, 0, 0, 0, 0, 0, 0],
    [104, 326, 80, 2, 0, 164, 157, 138, 194, 317, 43, 249, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [30, 1, 115, 355, 179, 185, 0, 178, 18, 353, 48, 0, 14, 337, 0, 0, 0, 0, 0, 0, 0, 0],
    [277, 282, 0, 0, 209, 41, 116, 189, 286, 224, 250, 93, 224, 287, 0, 0, 0, 0, 0, 0, 0, 0],
    [258, 292, 0, 0, 232, 55, 79, 158, 189, 292, 278, 273, 189, 37, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 260, 280, 10, 212, 141, 108, 257, 329, 249, 243, 220, 333, 0, 0, 0, 0, 0, 0, 0, 0],
    [269, 0, 57, 126, 202, 225, 0, 315, 80, 341, 90, 100, 38, 170, 0, 0, 0, 0, 0, 0, 0, 0],
    [286, 0, 75, 46, 59, 314, 194, 147, 41, 132, 0, 273, 0, 314, 0, 0, 0, 0, 0, 0, 0, 0],
    [258, 0, 219, 223, 0, 192, 96, 91, 32, 318, 268, 8, 0, 318, 0, 0, 0, 0, 0, 0, 0, 0],
    [23, 0, 13, 177, 109, 98, 0, 143, 116, 146, 112, 218, 37, 218, 0, 0, 0, 0, 0, 0, 0, 0],
    [23, 0, 132, 356, 117, 108, 0, 137, 42, 362, 110, 83, 13, 12, 0, 0, 0, 0, 0, 0, 0, 0],
    [116, 123, 20, 0, 18, 190, 94, 201, 0, 141, 113, 157, 111, 100, 0, 0, 0, 0, 0, 0, 0, 0],
    [114, 110, 91, 362, 27, 0, 0, 126, 7, 365, 110, 78, 17, 6, 0, 0, 0, 0, 0, 0, 0, 0],
    [36, 0, 148, 364, 117, 107, 0, 135, 59, 369, 107, 78, 5, 98, 0, 0, 0, 0, 0, 0, 0, 0],
    [15, 197, 35, 0, 118, 118, 132, 234, 45, 236, 0, 158, 21, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [56, 2, 39, 201, 142, 223, 0, 137, 113, 105, 67, 226, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [59, 1, 115, 97, 109, 368, 0, 132, 192, 365, 70, 0, 0, 121, 0, 0, 0, 0, 0, 0, 0, 0],
    [131, 116, 55, 364, 83, 0, 0, 128, 125, 363, 133, 254, 0, 114, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 125, 126, 222, 77, 0, 18, 182, 131, 126, 39, 212, 134, 183, 0, 0, 0, 0, 0, 0, 0, 0],
    [146, 149, 53, 0, 75, 248, 7, 128, 0, 246, 140, 169, 91, 237, 0, 0, 0, 0, 0, 0, 0, 0],
    [83, 370, 60, 0, 149, 140, 1, 327, 12, 128, 3, 370, 0, 299, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 134, 38, 365, 30, 0, 126, 142, 13, 280, 125, 365, 11, 65, 0, 0, 0, 0, 0, 0, 0, 0],
    [126, 142, 29, 0, 0, 133, 94, 232, 11, 223, 2, 163, 10, 64, 0, 0, 0, 0, 0, 0, 0, 0],
    [179, 45, 0, 0, 218, 130, 98, 115, 137, 133, 154, 18, 192, 137, 0, 0, 0, 0, 0, 0, 0, 0],
    [185, 254, 5, 0, 158, 38, 82, 154, 188, 66, 0, 17, 174, 49, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 299, 327, 169, 31, 300, 193, 92, 156, 201, 59, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [153, 152, 0, 0, 287, 147, 185, 17, 118, 142, 217, 45, 212, 159, 0, 0, 0, 0, 0, 0, 0, 0],
    [132, 169, 180, 0, 0, 193, 44, 105, 165, 138, 84, 197, 65, 75, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 269, 150, 2, 135, 122, 27, 67, 0, 107, 119, 144, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [292, 0, 0, 356, 161, 67, 280, 126, 77, 356, 0, 289, 131, 106, 0, 0, 0, 0, 0, 0, 0, 0],
    [225, 134, 261, 0, 116, 54, 0, 175, 152, 172, 79, 87, 200, 153, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 371, 10, 8, 97, 369, 87, 63, 0, 109, 98, 351, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [103, 282, 89, 69, 9, 2, 0, 114, 31, 286, 20, 0, 5, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [16, 0, 95, 139, 0, 125, 98, 90, 13, 170, 81, 175, 6, 35, 0, 0, 0, 0, 0, 0, 0, 0],
    [18, 180, 7, 0, 96, 132, 0, 136, 96, 82, 77, 177, 98, 104, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 9, 80, 295, 99, 121, 0, 130, 15, 297, 98, 77, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 125, 21, 368, 15, 0, 101, 74, 97, 365, 5, 10, 105, 337, 0, 0, 0, 0, 0, 0, 0, 0],
    [34, 369, 8, 8, 99, 88, 125, 367, 0, 115, 91, 61, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [112, 271, 10, 8, 98, 99, 0, 124, 43, 273, 20, 0, 93, 75, 0, 0, 0, 0, 0, 0, 0, 0],
    [95, 80, 16, 0, 0, 122, 30, 191, 95, 191, 98, 131, 9, 151, 0, 0, 0, 0, 0, 0, 0, 0],
    [99, 99, 21, 151, 65, 0, 113, 185, 0, 103, 57, 187, 34, 167, 0, 0, 0, 0, 0, 0, 0, 0],
    [60, 0, 142, 283, 11, 135, 99, 91, 80, 284, 0, 109, 0, 97, 0, 0, 0, 0, 0, 0, 0, 0],
    [157, 343, 51, 1, 0, 99, 69, 359, 98, 83, 61, 0, 157, 360, 0, 0, 0, 0, 0, 0, 0, 0],
    [106, 96, 92, 366, 21, 0, 0, 102, 2, 243, 5, 366, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [107, 105, 9, 261, 29, 0, 0, 106, 81, 284, 4, 228, 14, 285, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 196, 18, 0, 110, 114, 0, 117, 73, 194, 31, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0],
    [131, 26, 225, 168, 0, 0, 66, 103, 176, 173, 149, 48, 220, 148, 0, 0, 0, 0, 0, 0, 0, 0],
    [142, 46, 203, 277, 0, 15, 205, 165, 58, 117, 123, 27, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [185, 3, 0, 167, 160, 100, 98, 43, 80, 169, 179, 81, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [230, 0, 8, 294, 145, 29, 217, 70, 0, 185, 58, 245, 166, 17, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 8, 170, 213, 161, 0, 37, 321, 132, 323, 0, 27, 175, 190, 0, 0, 0, 0, 0, 0, 0, 0],
    [9, 42, 139, 227, 0, 216, 79, 0, 136, 314, 40, 316, 135, 174, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 0, 90, 302, 149, 216, 154, 11, 0, 158, 1, 301, 162, 175, 34, 16, 0, 0, 0, 0, 0, 0],
    [260, 157, 0, 97, 285, 319, 110, 0, 191, 283, 293, 237, 224, 301, 0, 0, 0, 0, 0, 0, 0, 0],
    [59, 0, 242, 129, 159, 237, 278, 181, 0, 57, 277, 271, 121, 216, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 115, 263, 303, 268, 114, 81, 0, 305, 310, 313, 235, 239, 78, 0, 0, 0, 0, 0, 0, 0, 0],
    [173, 0, 55, 331, 281, 94, 161, 228, 22, 191, 0, 331, 0, 232, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 284, 266, 32, 53, 122, 140, 226, 208, 0, 79, 297, 169, 192, 0, 0, 0, 0, 0, 0, 0, 0],
    [194, 0, 1, 243, 208, 251, 299, 88, 0, 326, 47, 326, 169, 277, 0, 0, 0, 0, 0, 0, 0, 0],
    [43, 223, 100, 0, 0, 21, 126, 120, 116, 223, 126, 173, 35, 203, 0, 0, 0, 0, 0, 0, 0, 0],
    [102, 2, 135, 366, 3, 21, 124, 111, 49, 369, 90, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0],
    [120, 120, 0, 13, 22, 193, 97, 0, 97, 198, 14, 150, 111, 170, 0, 0, 0, 0, 0, 0, 0, 0],
    [89, 368, 2, 9, 100, 2, 121, 111, 4, 369, 0, 25, 90, 0, 27, 7, 0, 0, 0, 0, 0, 0],
    [0, 15, 114, 358, 96, 0, 4, 203, 106, 115, 30, 360, 11, 248, 0, 0, 0, 0, 0, 0, 0, 0],
    [17, 214, 3, 8, 119, 153, 106, 0, 116, 188, 0, 30, 105, 219, 0, 0, 0, 0, 0, 0, 0, 0],
    [114, 0, 147, 170, 0, 219, 26, 4, 96, 221, 152, 152, 135, 186, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 355, 117, 5, 159, 145, 35, 0, 80, 360, 3, 286, 157, 153, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 2, 279, 136, 142, 96, 0, 13, 351, 95, 353, 0, 252, 0, 0, 0, 0, 0, 0, 0, 0],
    [96, 0, 116, 186, 0, 217, 4, 9, 72, 222, 133, 154, 108, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [19, 21, 109, 209, 31, 222, 95, 0, 0, 159, 117, 155, 10, 195, 0, 0, 0, 0, 0, 0, 0, 0],
    [41, 363, 6, 33, 5, 205, 101, 0, 118, 359, 115, 144, 0, 157, 0, 0, 0, 0, 0, 0, 0, 0],
    [89, 0, 80, 371, 0, 175, 164, 369, 31, 39, 9, 212, 4, 138, 0, 0, 0, 0, 0, 0, 0, 0],
    [85, 0, 47, 255, 134, 252, 24, 26, 0, 172, 115, 135, 14, 215, 0, 0, 0, 0, 0, 0, 0, 0],
    [154, 192, 77, 0, 0, 65, 187, 123, 187, 179, 119, 188, 172, 100, 0, 0, 0, 0, 0, 0, 0, 0],
    [207, 300, 0, 77, 192, 102, 210, 164, 65, 0, 202, 123, 163, 65, 0, 0, 0, 0, 0, 0, 0, 0],
    [196, 95, 277, 357, 0, 82, 54, 0, 299, 246, 163, 63, 299, 350, 0, 0, 0, 0, 0, 0, 0, 0],
    [40, 0, 282, 171, 192, 57, 183, 168, 0, 75, 305, 155, 143, 157, 0, 0, 0, 0, 0, 0, 0, 0],
    [203, 70, 0, 142, 143, 0, 118, 142, 21, 103, 52, 185, 0, 185, 0, 0, 0, 0, 0, 0, 0, 0],
    [16, 112, 195, 48, 2, 302, 132, 0, 38, 85, 0, 145, 196, 61, 0, 0, 0, 0, 0, 0, 0, 0],
    [132, 100, 248, 0, 0, 361, 313, 51, 0, 284, 67, 361, 115, 118, 0, 0, 0, 0, 0, 0, 0, 0],
    [195, 0, 82, 167, 162, 133, 252, 68, 0, 164, 47, 100, 126, 152, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 10, 84, 358, 66, 0, 88, 88, 2, 359, 0, 19, 88, 76, 0, 0, 0, 0, 0, 0, 0, 0],
    [82, 270, 1, 11, 89, 86, 73, 2, 16, 271, 0, 24, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [74, 0, 89, 97, 0, 5, 12, 168, 72, 166, 85, 134, 8, 150, 0, 0, 0, 0, 0, 0, 0, 0],
    [76, 0, 77, 164, 0, 8, 96, 92, 17, 177, 91, 131, 69, 176, 0, 0, 0, 0, 0, 0, 0, 0],
    [103, 88, 75, 270, 78, 0, 1, 12, 15, 270, 98, 129, 0, 23, 23, 4, 0, 0, 0, 0, 0, 0],
    [0, 355, 104, 4, 86, 354, 29, 11, 118, 82, 89, 342, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 351, 78, 1, 3, 12, 85, 354, 88, 81, 88, 95, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [75, 0, 31, 270, 8, 157, 0, 11, 101, 268, 90, 91, 15, 195, 0, 0, 0, 0, 0, 0, 0, 0],
    [93, 120, 76, 0, 0, 6, 11, 164, 81, 167, 93, 92, 90, 142, 0, 0, 0, 0, 0, 0, 0, 0],
    [93, 134, 9, 30, 11, 157, 52, 0, 0, 117, 84, 95, 94, 159, 0, 0, 0, 0, 0, 0, 0, 0],
    [61, 272, 11, 40, 54, 0, 86, 99, 0, 114, 125, 272, 7, 153, 0, 0, 0, 0, 0, 0, 0, 0],
    [58, 0, 133, 356, 12, 47, 46, 371, 0, 107, 85, 97, 126, 314, 0, 0, 0, 0, 0, 0, 0, 0],
    [87, 12, 0, 376, 116, 112, 12, 1, 85, 375, 0, 363, 21, 0, 19, 19, 0, 0, 0, 0, 0, 0],
    [12, 256, 87, 2, 120, 117, 0, 9, 76, 261, 9, 241, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 7, 111, 137, 87, 0, 16, 184, 77, 184, 0, 18, 119, 115, 0, 0, 0, 0, 0, 0, 0, 0],
    [128, 138, 0, 52, 233, 135, 155, 54, 38, 0, 187, 148, 103, 128, 0, 0, 0, 0, 0, 0, 0, 0],
    [38, 0, 216, 277, 139, 55, 219, 175, 0, 53, 116, 38, 151, 69, 0, 0, 0, 0, 0, 0, 0, 0],
    [200, 44, 94, 132, 0, 133, 66, 57, 153, 0, 121, 118, 50, 72, 0, 0, 0, 0, 0, 0, 0, 0],
    [187, 0, 0, 141, 233, 46, 110, 40, 3, 229, 91, 55, 233, 37, 0, 0, 0, 0, 0, 0, 0, 0],
    [60, 42, 232, 347, 190, 0, 0, 208, 242, 130, 143, 348, 241, 218, 43, 0, 0, 71, 0, 0, 0, 0],
    [128, 0, 209, 326, 0, 184, 205, 214, 188, 117, 65, 42, 126, 338, 31, 0, 0, 54, 0, 0, 0, 0],
    [24, 26, 175, 346, 192, 144, 0, 175, 146, 0, 89, 346, 191, 117, 38, 0, 0, 43, 0, 0, 0, 0],
    [247, 110, 0, 99, 65, 0, 295, 297, 49, 249, 184, 50, 296, 195, 0, 0, 16, 76, 0, 0, 0, 0],
    [96, 241, 63, 0, 304, 309, 0, 100, 181, 62, 306, 235, 286, 316, 0, 0, 27, 72, 0, 0, 0, 0],
    [39, 0, 271, 260, 42, 192, 172, 51, 272, 156, 0, 53, 251, 120, 0, 0, 22, 69, 0, 0, 0, 0],
    [0, 267, 297, 69, 53, 37, 209, 0, 226, 172, 70, 322, 0, 322, 73, 0, 0, 6, 0, 0, 0, 0],
    [0, 295, 218, 0, 77, 41, 202, 139, 267, 27, 78, 295, 0, 265, 66, 0, 0, 18, 0, 0, 0, 0],
    [273, 83, 0, 319, 79, 54, 0, 184, 216, 0, 199, 173, 27, 319, 48, 9, 0, 0, 0, 0, 0, 0],
    [0, 130, 139, 0, 170, 80, 171, 134, 57, 16, 93, 213, 161, 213, 32, 0, 0, 46, 0, 0, 0, 0],
    [64, 19, 108, 358, 0, 118, 140, 0, 192, 359, 172, 74, 192, 347, 31, 0, 0, 39, 0, 0, 0, 0],
    [0, 108, 168, 148, 150, 0, 68, 8, 176, 86, 77, 187, 153, 184, 38, 0, 0, 40, 0, 0, 0, 0],
    [181, 80, 172, 354, 70, 16, 0, 114, 150, 0, 95, 358, 155, 359, 33, 0, 0, 38, 0, 0, 0, 0],
    [0, 121, 196, 360, 152, 3, 71, 20, 110, 363, 179, 83, 141, 0, 35, 0, 0, 41, 0, 0, 0, 0],
    [71, 10, 179, 156, 0, 124, 157, 0, 184, 94, 82, 206, 165, 207, 41, 0, 0, 45, 0, 0, 0, 0],
    [153, 180, 123, 0, 0, 130, 58, 212, 70, 25, 151, 93, 144, 219, 28, 0, 0, 49, 0, 0, 0, 0],
    [91, 364, 78, 35, 0, 142, 136, 0, 165, 361, 159, 86, 146, 366, 0, 0, 0, 0, 0, 0, 0, 0],
    [120, 0, 196, 353, 0, 135, 158, 86, 70, 30, 112, 357, 131, 1, 25, 0, 0, 43, 0, 0, 0, 0],
    [140, 0, 55, 198, 175, 232, 0, 122, 169, 97, 75, 18, 93, 236, 32, 0, 0, 39, 0, 0, 0, 0],
    [70, 221, 123, 0, 0, 128, 161, 89, 48, 9, 143, 221, 161, 108, 30, 0, 0, 45, 0, 0, 0, 0],
    [49, 4, 174, 348, 162, 87, 120, 2, 0, 96, 91, 352, 110, 0, 32, 0, 0, 26, 0, 0, 0, 0],
    [51, 0, 131, 353, 163, 91, 0, 92, 122, 0, 52, 353, 163, 105, 33, 0, 0, 20, 0, 0, 0, 0],
    [0, 118, 161, 110, 43, 8, 130, 206, 121, 0, 57, 225, 124, 223, 34, 0, 0, 31, 0, 0, 0, 0],
    [189, 99, 0, 64, 30, 204, 207, 216, 144, 48, 51, 0, 210, 143, 0, 0, 9, 65, 0, 0, 0, 0],
    [207, 110, 0, 71, 35, 215, 209, 249, 150, 41, 47, 0, 197, 91, 0, 0, 9, 65, 0, 0, 0, 0],
    [300, 308, 0, 73, 200, 75, 39, 213, 302, 198, 36, 0, 150, 32, 0, 0, 12, 67, 0, 0, 0, 0],
    [300, 230, 0, 68, 37, 212, 154, 41, 43, 0, 301, 169, 281, 240, 0, 0, 11, 68, 0, 0, 0, 0],
    [205, 8, 0, 176, 176, 144, 50, 0, 253, 71, 124, 171, 79, 185, 65, 10, 0, 0, 0, 0, 0, 0],
    [221, 48, 3, 322, 24, 8, 0, 163, 171, 1, 162, 0, 219, 38, 56, 0, 0, 1, 0, 0, 0, 0],
    [0, 343, 250, 1, 100, 1, 0, 254, 300, 42, 55, 343, 111, 0, 62, 1, 0, 0, 0, 0, 0, 0],
    [90, 0, 198, 152, 0, 182, 279, 91, 144, 171, 236, 25, 92, 184, 63, 17, 0, 0, 0, 0, 0, 0],
    [119, 157, 103, 0, 0, 93, 131, 120, 51, 10, 133, 64, 66, 158, 26, 0, 0, 31, 0, 0, 0, 0],
    [0, 93, 105, 0, 138, 251, 139, 104, 50, 12, 75, 251, 137, 62, 24, 0, 0, 31, 0, 0, 0, 0],
    [148, 348, 53, 11, 144, 63, 0, 89, 108, 0, 60, 363, 147, 363, 26, 0, 0, 29, 0, 0, 0, 0],
    [120, 188, 66, 5, 0, 89, 143, 134, 126, 0, 65, 190, 154, 82, 55, 3, 33, 0, 0, 31, 0, 0],
    [127, 2, 130, 264, 0, 97, 155, 82, 63, 10, 69, 264, 117, 0, 30, 0, 0, 32, 0, 0, 0, 0],
    [122, 0, 44, 367, 0, 92, 130, 364, 158, 80, 67, 11, 132, 1, 33, 0, 0, 32, 0, 0, 0, 0],
    [0, 93, 67, 360, 129, 2, 153, 354, 64, 12, 152, 79, 117, 0, 31, 0, 0, 32, 0, 0, 0, 0],
    [129, 0, 0, 89, 146, 285, 154, 86, 66, 8, 75, 285, 136, 4, 35, 0, 0, 32, 0, 0, 0, 0],
    [130, 0, 155, 86, 63, 6, 0, 93, 149, 137, 69, 186, 132, 185, 33, 0, 0, 34, 0, 0, 0, 0],
    [123, 109, 55, 18, 0, 102, 131, 184, 97, 0, 117, 68, 74, 187, 21, 0, 0, 38, 0, 0, 0, 0],
    [89, 0, 147, 261, 0, 110, 84, 265, 119, 67, 54, 24, 98, 0, 20, 0, 0, 43, 0, 0, 0, 0],
    [160, 364, 103, 0, 0, 113, 69, 367, 127, 64, 60, 27, 160, 345, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 59, 35, 343, 124, 74, 92, 7, 121, 344, 42, 0, 125, 85, 24, 0, 0, 12, 0, 0, 0, 0],
    [43, 1, 63, 250, 107, 2, 0, 75, 131, 82, 127, 251, 52, 0, 27, 0, 0, 22, 0, 0, 0, 0],
    [113, 168, 44, 5, 9, 117, 118, 0, 140, 93, 46, 180, 0, 91, 33, 0, 0, 35, 0, 0, 0, 0],
    [16, 139, 191, 163, 108, 33, 32, 0, 0, 42, 139, 66, 153, 171, 0, 0, 1, 42, 0, 0, 0, 0],
    [232, 145, 24, 0, 29, 146, 232, 227, 0, 49, 145, 40, 111, 14, 0, 0, 10, 45, 0, 0, 0, 0],
    [113, 0, 244, 45, 0, 174, 194, 105, 208, 7, 80, 174, 0, 161, 48, 8, 0, 0, 0, 0, 0, 0],
    [0, 169, 244, 52, 118, 0, 2, 274, 193, 104, 213, 13, 245, 44, 42, 9, 0, 0, 0, 0, 0, 0],
    [0, 55, 50, 267, 240, 72, 203, 344, 90, 0, 168, 11, 111, 344, 82, 15, 43, 0, 0, 5, 0, 0],
    [0, 52, 19, 234, 180, 329, 67, 0, 127, 9, 102, 331, 175, 67, 72, 5, 37, 0, 0, 15, 0, 0],
    [0, 29, 38, 326, 172, 82, 52, 0, 118, 21, 138, 338, 17, 227, 67, 63, 39, 0, 0, 5, 0, 0],
    [142, 233, 29, 38, 183, 0, 288, 296, 0, 127, 90, 3, 300, 202, 65, 0, 25, 13, 0, 40, 0, 0],
    [151, 0, 287, 293, 0, 122, 68, 5, 13, 43, 287, 213, 253, 293, 63, 17, 0, 0, 13, 56, 0, 0],
    [112, 205, 26, 11, 273, 288, 154, 32, 273, 175, 0, 79, 80, 0, 7, 0, 0, 44, 0, 0, 0, 0],
    [0, 320, 237, 19, 43, 147, 278, 156, 143, 0, 93, 325, 274, 69, 44, 56, 30, 22, 0, 0, 0, 0],
    [143, 0, 0, 299, 38, 131, 222, 3, 241, 110, 87, 300, 249, 42, 40, 3, 0, 0, 66, 25, 0, 0],
    [243, 152, 206, 14, 0, 239, 28, 325, 239, 70, 151, 0, 44, 159, 10, 85, 27, 23, 0, 0, 0, 0],
    [0, 28, 125, 367, 139, 47, 7, 145, 51, 0, 38, 367, 100, 10, 52, 11, 27, 0, 0, 2, 0, 0],
    [47, 0, 24, 168, 148, 47, 0, 29, 115, 223, 102, 9, 51, 223, 55, 11, 29, 0, 0, 4, 0, 0],
    [28, 167, 118, 10, 0, 34, 48, 204, 168, 51, 128, 222, 60, 0, 59, 12, 30, 0, 0, 4, 0, 0],
    [58, 0, 52, 359, 166, 49, 18, 162, 0, 33, 139, 360, 117, 8, 62, 12, 32, 0, 0, 5, 0, 0],
    [14, 162, 55, 0, 64, 360, 0, 32, 148, 45, 149, 361, 107, 8, 56, 11, 29, 0, 0, 4, 0, 0],
    [0, 29, 49, 224, 114, 18, 17, 180, 62, 0, 120, 224, 152, 63, 56, 15, 31, 0, 0, 3, 0, 0],
    [11, 175, 0, 35, 123, 48, 117, 225, 91, 7, 49, 0, 56, 227, 53, 6, 28, 0, 0, 15, 0, 0],
    [129, 347, 47, 0, 0, 38, 3, 171, 126, 33, 68, 348, 90, 1, 57, 5, 29, 0, 0, 19, 0, 0],
    [157, 354, 0, 43, 9, 170, 70, 0, 78, 357, 112, 30, 34, 4, 52, 8, 25, 0, 0, 18, 0, 0],
    [46, 0, 20, 205, 120, 47, 116, 229, 3, 31, 0, 167, 89, 7, 56, 1, 30, 0, 0, 6, 0, 0],
    [104, 225, 0, 18, 148, 73, 27, 225, 49, 0, 108, 23, 11, 174, 54, 30, 32, 4, 0, 0, 0, 0],
    [5, 160, 70, 0, 26, 344, 141, 70, 0, 0, 109, 346, 15, 262, 53, 47, 33, 6, 0, 0, 0, 0],
    [136, 65, 0, 361, 15, 21, 82, 363, 101, 19, 64, 0, 54, 0, 45, 64, 28, 0, 0, 14, 0, 0],
    [12, 190, 88, 13, 92, 173, 0, 35, 123, 56, 43, 0, 81, 193, 46, 56, 30, 0, 0, 20, 0, 0],
    [65, 7, 231, 196, 0, 103, 123, 179, 137, 0, 15, 36, 234, 128, 49, 0, 18, 11, 0, 35, 0, 0],
    [221, 115, 64, 15, 0, 119, 124, 199, 228, 248, 142, 0, 13, 52, 53, 0, 20, 16, 0, 43, 0, 0],
    [4, 136, 308, 300, 133, 0, 55, 20, 310, 189, 9, 62, 0, 127, 48, 0, 17, 19, 0, 48, 0, 0],
    [303, 194, 36, 27, 4, 131, 103, 0, 141, 180, 0, 71, 305, 121, 38, 0, 10, 18, 0, 48, 0, 0],
    [34, 86, 208, 118, 132, 0, 0, 163, 109, 166, 186, 26, 211, 63, 31, 45, 21, 19, 0, 0, 0, 0],
    [0, 234, 188, 54, 0, 67, 180, 113, 113, 0, 171, 20, 29, 48, 36, 9, 0, 0, 35, 42, 0, 0],
    [260, 18, 19, 368, 95, 95, 297, 116, 0, 263, 200, 0, 291, 52, 36, 45, 24, 17, 0, 0, 0, 0],
    [277, 32, 135, 203, 0, 202, 84, 83, 289, 148, 219, 0, 298, 81, 19, 24, 0, 0, 24, 55, 0, 0],
    [111, 35, 111, 370, 0, 21, 31, 368, 4, 115, 38, 0, 77, 8, 44, 9, 23, 0, 0, 1, 0, 0],
    [0, 20, 90, 293, 113, 42, 0, 118, 19, 293, 79, 9, 42, 0, 43, 10, 23, 0, 0, 0, 0, 0],
    [85, 12, 7, 139, 0, 21, 121, 47, 89, 197, 44, 0, 33, 197, 47, 12, 25, 0, 0, 1, 0, 0],
    [129, 42, 87, 180, 0, 21, 22, 147, 54, 0, 108, 129, 41, 182, 50, 11, 25, 0, 0, 2, 0, 0],
    [97, 269, 41, 0, 18, 135, 134, 38, 0, 28, 53, 273, 90, 5, 50, 8, 26, 0, 0, 3, 0, 0],
    [0, 26, 100, 367, 134, 36, 43, 0, 44, 369, 88, 4, 118, 336, 51, 8, 26, 0, 0, 4, 0, 0],
    [42, 0, 115, 365, 127, 38, 8, 134, 0, 26, 31, 365, 87, 7, 51, 9, 26, 0, 0, 1, 0, 0],
    [0, 21, 27, 282, 127, 53, 0, 135, 96, 282, 47, 0, 90, 12, 48, 13, 25, 0, 0, 1, 0, 0],
    [0, 148, 51, 0, 129, 56, 90, 212, 2, 18, 29, 211, 96, 16, 47, 14, 26, 1, 0, 0, 0, 0],
    [29, 210, 39, 0, 126, 33, 10, 184, 0, 37, 122, 221, 81, 0, 56, 15, 29, 0, 0, 15, 0, 0],
    [0, 142, 69, 0, 103, 256, 2, 27, 49, 256, 101, 26, 34, 0, 49, 0, 25, 6, 0, 14, 0, 0],
    [104, 23, 107, 367, 8, 31, 0, 136, 67, 0, 23, 368, 108, 320, 50, 0, 25, 13, 0, 21, 0, 0],
    [109, 209, 50, 0, 0, 153, 113, 45, 19, 189, 6, 25, 86, 7, 52, 0, 27, 0, 0, 8, 0, 0],
    [80, 209, 45, 0, 127, 75, 11, 224, 1, 11, 0, 159, 91, 23, 47, 45, 29, 0, 0, 6, 0, 0],
    [24, 323, 0, 14, 141, 68, 4, 159, 99, 322, 99, 21, 48, 0, 49, 35, 29, 0, 0, 14, 0, 0],
    [264, 152, 0, 90, 102, 148, 38, 14, 95, 0, 239, 177, 7, 38, 35, 0, 11, 11, 0, 32, 0, 0],
    [91, 0, 234, 243, 0, 91, 233, 163, 6, 43, 37, 14, 235, 184, 34, 0, 12, 13, 0, 33, 0, 0],
    [206, 85, 94, 40, 52, 169, 165, 0, 0, 148, 211, 49, 199, 20, 13, 13, 0, 0, 15, 32, 0, 0],
    [95, 40, 2, 234, 211, 45, 161, 0, 0, 149, 207, 78, 198, 18, 15, 11, 0, 0, 20, 29, 0, 0],
    [235, 337, 45, 46, 262, 61, 0, 182, 121, 0, 196, 10, 145, 337, 111, 14, 72, 0, 32, 7, 0, 70],
    [239, 46, 0, 179, 228, 326, 55, 46, 116, 0, 142, 328, 185, 0, 100, 0, 62, 1, 25, 15, 0, 68],
    [0, 120, 151, 17, 149, 204, 47, 21, 106, 0, 188, 60, 86, 204, 78, 17, 56, 0, 26, 4, 0, 45],
    [198, 59, 0, 128, 150, 201, 43, 25, 155, 15, 106, 0, 89, 202, 81, 13, 55, 0, 24, 1, 0, 41],
    [45, 28, 165, 341, 198, 49, 0, 128, 100, 0, 85, 341, 153, 9, 81, 11, 54, 0, 25, 4, 0, 47],
    [95, 0, 92, 354, 0, 130, 191, 47, 177, 355, 43, 29, 147, 8, 77, 11, 51, 0, 22, 6, 0, 48],
    [189, 60, 41, 25, 86, 209, 0, 127, 100, 0, 151, 209, 147, 15, 80, 16, 55, 0, 25, 4, 0, 47],
    [155, 12, 63, 195, 98, 0, 0, 120, 162, 190, 193, 60, 51, 25, 78, 9, 52, 0, 22, 3, 0, 41],
    [0, 135, 158, 2, 187, 356, 53, 34, 197, 34, 100, 0, 105, 358, 84, 0, 52, 8, 22, 17, 0, 58],
    [109, 358, 76, 3, 0, 150, 156, 30, 192, 356, 41, 43, 125, 0, 67, 0, 42, 40, 12, 50, 0, 108],
    [86, 0, 0, 134, 161, 204, 174, 53, 61, 204, 140, 9, 44, 29, 77, 0, 46, 7, 13, 6, 0, 62],
    [163, 58, 20, 22, 134, 213, 0, 134, 65, 0, 116, 16, 62, 219, 73, 45, 47, 0, 16, 8, 0, 70],
    [203, 339, 92, 0, 0, 158, 227, 68, 162, 17, 30, 28, 112, 339, 100, 30, 63, 0, 28, 15, 0, 68],
    [163, 66, 77, 356, 20, 16, 116, 22, 0, 113, 67, 0, 160, 357, 77, 47, 52, 0, 20, 2, 0, 46],
    [150, 77, 34, 361, 17, 14, 0, 118, 114, 361, 65, 0, 108, 26, 49, 59, 30, 2, 0, 0, 0, 0],
    [60, 0, 99, 219, 142, 69, 0, 138, 27, 222, 13, 24, 103, 21, 45, 60, 30, 0, 0, 8, 0, 0],
    [0, 71, 219, 130, 133, 0, 34, 176, 68, 1, 24, 19, 215, 214, 52, 0, 12, 0, 0, 27, 11, 81],
    [0, 88, 235, 129, 150, 1, 34, 202, 246, 297, 23, 28, 76, 0, 59, 0, 19, 7, 0, 35, 11, 95],
    [0, 79, 298, 232, 286, 364, 134, 0, 31, 183, 19, 27, 68, 0, 53, 0, 16, 7, 0, 34, 11, 90],
    [0, 77, 287, 190, 67, 0, 121, 193, 42, 176, 18, 22, 130, 2, 52, 0, 13, 3, 0, 31, 18, 86],
    [222, 109, 71, 5, 0, 162, 214, 28, 168, 0, 85, 163, 229, 64, 60, 17, 41, 0, 0, 0, 65, 40],
    [167, 20, 28, 20, 182, 110, 0, 166, 124, 0, 57, 177, 189, 63, 60, 16, 42, 0, 0, 6, 66, 39],
    [41, 9, 222, 82, 3, 257, 164, 0, 198, 139, 0, 158, 212, 35, 74, 20, 51, 1, 0, 0, 69, 51],
    [39, 22, 274, 334, 27, 237, 191, 13, 0, 93, 108, 0, 291, 237, 71, 0, 30, 5, 0, 29, 6, 101],
    [246, 0, 3, 335, 289, 136, 124, 7, 309, 82, 0, 221, 298, 35, 69, 23, 52, 1, 0, 0, 69, 51],
    [146, 34, 110, 353, 37, 19, 0, 95, 29, 366, 76, 0, 115, 7, 60, 8, 40, 0, 17, 1, 0, 33],
    [152, 38, 123, 254, 39, 17, 0, 95, 119, 7, 58, 255, 79, 0, 63, 9, 42, 0, 20, 2, 0, 35],
    [0, 96, 158, 43, 41, 16, 122, 182, 125, 10, 69, 182, 82, 0, 66, 10, 45, 0, 21, 1, 0, 35],
    [117, 7, 75, 184, 35, 20, 0, 101, 130, 184, 152, 39, 76, 0, 62, 11, 40, 0, 17, 5, 0, 38],
    [0, 104, 80, 0, 135, 304, 157, 40, 38, 21, 61, 305, 124, 8, 65, 13, 43, 0, 19, 5, 0, 41],
    [123, 7, 130, 371, 38, 23, 157, 39, 0, 107, 80, 0, 49, 373, 65, 12, 44, 0, 18, 4, 0, 41],
    [127, 6, 141, 357, 0, 100, 47, 19, 63, 358, 161, 35, 88, 0, 68, 8, 47, 0, 22, 3, 0, 37],
    [87, 0, 76, 283, 0, 101, 163, 40, 144, 283, 47, 19, 129, 8, 68, 10, 46, 0, 21, 3, 0, 35],
    [132, 11, 0, 95, 130, 167, 88, 0, 48, 16, 165, 44, 73, 182, 71, 12, 49, 0, 24, 0, 0, 34],
    [253, 324, 0, 67, 158, 36, 16, 195, 263, 206, 87, 0, 31, 7, 20, 0, 0, 33, 12, 99, 0, 0],
    [129, 2, 0, 121, 153, 208, 40, 28, 81, 0, 159, 34, 88, 209, 70, 0, 42, 15, 16, 23, 0, 60],
    [42, 22, 64, 272, 149, 29, 0, 96, 133, 273, 77, 0, 120, 3, 65, 0, 40, 8, 16, 15, 0, 43],
    [77, 0, 142, 340, 0, 104, 157, 26, 42, 25, 59, 344, 126, 2, 67, 0, 40, 8, 15, 14, 0, 47],
    [157, 43, 0, 97, 71, 211, 83, 0, 126, 210, 46, 19, 130, 10, 68, 2, 43, 3, 15, 0, 0, 40],
    [108, 147, 74, 0, 0, 97, 133, 55, 33, 206, 90, 206, 17, 9, 59, 43, 44, 0, 18, 13, 0, 48],
    [89, 22, 87, 311, 0, 93, 13, 12, 15, 311, 124, 63, 56, 0, 60, 49, 44, 0, 18, 6, 0, 44],
    [164, 165, 15, 25, 29, 141, 103, 0, 180, 129, 0, 65, 54, 1, 38, 0, 15, 8, 0, 25, 11, 62],
    [196, 204, 47, 9, 0, 90, 101, 0, 41, 167, 12, 40, 198, 130, 37, 0, 14, 13, 0, 34, 17, 73],
    [92, 7, 2, 283, 231, 49, 0, 178, 176, 0, 222, 82, 217, 18, 62, 10, 37, 0, 0, 2, 74, 28],
    [249, 113, 99, 11, 0, 201, 198, 0, 244, 27, 92, 203, 259, 68, 60, 19, 43, 0, 0, 2, 64, 42],
    [85, 0, 270, 328, 0, 114, 79, 228, 171, 0, 14, 49, 283, 230, 71, 0, 37, 11, 0, 31, 37, 100],
    [32, 69, 226, 9, 104, 324, 287, 128, 0, 323, 149, 0, 274, 58, 108, 46, 89, 16, 58, 0, 0, 28],
    [252, 97, 0, 301, 143, 0, 38, 64, 211, 0, 97, 302, 251, 36, 125, 7, 86, 5, 50, 0, 0, 25],
    [184, 0, 0, 309, 278, 163, 91, 78, 248, 29, 81, 328, 277, 86, 72, 63, 83, 7, 42, 3, 0, 0],
    [0, 123, 52, 24, 88, 352, 190, 55, 108, 0, 172, 352, 155, 15, 76, 13, 54, 0, 25, 4, 0, 42]
]
tlabel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
          5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
          5, 5, 5, 5]
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(tdata, tlabel)
