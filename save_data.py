import csv
import cv2

save_is_init = False

csvfile = None
csv_wr = None

def save_init():
    global save_is_init, csvfile, csv_wr
    save_is_init = True
    sample = 0
    csvfile = open('sample-'+str(sample)+'.csv', 'w', newline='')
    csv_wr = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_wr.writerow(['№', 'label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'a_x1', 'a_y1', 'a_x2', 'a_y2', 'a_x3', 'a_y3', 'a_x4', 'a_y4', 'filename'])


def save_data(cap, coords1, coords2, f, img):
    global save_is_init, csvfile, csv_wr
    if save_is_init:
        global csv_wr
        while len(coords1) < 7:
            coords1.append([0, 0])
        while len(coords2) < 4:
            coords2.append([0, 0])

        filename = 'captures/capture-no_' + str(cap) + '-f_' + str(f) + '.bmp'
        coords = [
            cap,
            f,
            coords1[0][0],
            coords1[0][1],
            coords1[1][0],
            coords1[1][1],
            coords1[2][0],
            coords1[2][1],
            coords1[3][0],
            coords1[3][1],
            coords1[4][0],
            coords1[4][1],
            coords1[5][0],
            coords1[5][1],
            coords1[6][0],
            coords1[6][1],
            coords2[0][0],
            coords2[0][1],
            coords2[1][0],
            coords2[1][1],
            coords2[2][0],
            coords2[2][1],
            coords2[3][0],
            coords2[3][1]
        ]
        coords = [int(x) for x in coords]
        coords.append(filename)
        csv_wr.writerow(coords)
        csvfile.flush()
        cv2.imwrite(filename, img)
        print("[INFO] Row added" + str(cap))
