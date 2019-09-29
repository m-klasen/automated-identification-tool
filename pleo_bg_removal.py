import cv2
from google.colab.patches import cv2_imshow

def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours ):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        i+=1

    return largest_area, largest_contour_index


path="datasets/pleophyllax224zoomed224"

for fold in sorted(os.listdir(path)):
  for file in os.listdir(path+"/"+fold):
    file_to_path=path+"/"+fold+"/"+file 
    save_to="datasets/pleophyllax224zoomed224x"+"/"+fold
    os.makedirs(save_to, exist_ok=True)
    image = cv2.imread(file_to_path,0)
    image_og = cv2.imread(file_to_path)


    hsv = cv2.cvtColor(image_og, cv2.COLOR_BGR2HSV)

    green = np.uint8([[[80, 125, 140]]])
    hsvGreen = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    lowerLimit = (0,60,60)
    upperLimit = (70,255,255)

    ret, thresh = cv2.threshold(image, 200, 255, 0)
    _, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    #cv2.drawContours(image_og, contours, -2, (0, 0, 255), 3)
    mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, cntsSorted, 1, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(image_og) # Extract out the object and place into output image
    out[mask == 255] = image_og[mask == 255]
    out[np.where(mask==0)] = [255,255,255]
    cv2_imshow(image_og)
    cv2_imshow(out)

    
    cv2.imwrite('datasets/pleophyllax224zoomed224x/'+fold+'/'+file[:-4]+"-fix"+".jpg",out)
