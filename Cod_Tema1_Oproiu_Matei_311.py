import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================================
# def show_image(title,image):
#     image=cv.resize(image,(0,0),fx=0.2,fy=0.2)
#     cv.imshow(title,image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# =============================================================================
    
#Function to display an image
def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')  
    plt.show()

#Function to find the corners
def find_color_values_fixed(frame):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_bound = np.array([0, 0, 0])   
    upper_bound = np.array([85, 255, 255])

    mask_table_hsv = cv.inRange(frame_hsv, lower_bound, upper_bound)

    res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)

    return res

#Function that returns the top left and bottom right points of the game board
def extrage_careu(image):
    hue=find_color_values_fixed(image)
    
    image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    image_m_blur=cv.medianBlur(hue, 5)
    #show_image('1.Median Blur',image_m_blur)
    
    image_g_blur=cv.GaussianBlur(image_m_blur, (3,3), 0)
    #show_image('2.Gaussian Blur',image_g_blur)
    
    image_sharpened=cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    #show_image('3.Sharpened Img',image_sharpened)
    
    _,thresh=cv.threshold(image_sharpened, 50, 255, cv.THRESH_BINARY)
    #show_image('4.Thresh Img',thresh)
    
    kernel=np.ones((3,3),np.uint8)
    thresh=cv.erode(thresh,kernel)
    #show_image('5.Eroded Image',thresh)
    
    edges=cv.Canny(thresh,200,400)
    #show_image('6.Canny', edges)
    
    contours,_=cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    top_left_point = None
    bottom_right_point = None
    
    #Loop through contours
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            for point in contours[i].squeeze():
                if top_left_point is None or point[0] + point[1] < top_left_point[0] + top_left_point[1]:
                    top_left_point = point

                if bottom_right_point is None or point[0] + point[1] > bottom_right_point[0] + bottom_right_point[1] :
                    bottom_right_point = point

    image_copy=cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)
    cv.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    cv.circle(image_copy,tuple(top_left_point),3,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right_point),3,(0,0,255),-1)
    #show_image('7.Contours',image_copy)
    
    return top_left_point,bottom_right_point
    
#Function that maps out game board to a 1400x1400 image
def transform(image,top_left,top_right,bottom_left,bottom_right):
    width=1400 
    height=1400
    
    image_transform=image.copy()
    
    puzzle=np.array([top_left,top_right,bottom_right,bottom_left],dtype='float32')
    
    destination_puzzle=np.array([[0,0],[width,0],[width,height],[0,height]],dtype='float32')
    
    M=cv.getPerspectiveTransform(puzzle,destination_puzzle)
    
    result=cv.warpPerspective(image_transform,M,(width,height))
    
    return result

#Function that count how many contours are in a game board square 
#to determine if we have a one digit number or a two digit number
def count_contours(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    img_with_contours = img.copy()
    cv.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    
    valid_contours = []
    min_area=200
    max_area=2000
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        if 10 <= w <= 70 and 10 <= h <= 70 and min_area <= area <= max_area:
            valid_contours.append((x, y, w, h))
            
    img_with_boxes = img.copy()
    for x, y, w, h in valid_contours:
        cv.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return len(valid_contours)

#Function that calculates all the possible pieces that can go onto a square
def calculate_possible_pieces(i,j):
    pieces=[]
    if m[i,j]==0 or m[i,j]=='2x' or m[i,j]=='3x':
        
        if [i-1,j] in visited and [i-2,j] in visited:
            pieces.append(m[i-1,j]+m[i-2,j])
            pieces.append(abs(m[i-1,j]-m[i-2,j]))
            pieces.append(m[i-1,j]*m[i-2,j])
            if m[i-2,j]!=0 and m[i-1,j] % m[i-2,j] == 0:
                pieces.append(m[i-1,j]//m[i-2,j])
            if m[i-1,j]!=0 and m[i-2,j] % m[i-1,j] == 0:
                pieces.append(m[i-2,j]//m[i-1,j])
                
        if [i,j+1] in visited and [i,j+2] in visited:
            pieces.append(m[i,j+1]+m[i,j+2])
            pieces.append(abs(m[i,j+1]-m[i,j+2]))
            pieces.append(m[i,j+1]*m[i,j+2])
            if m[i,j+2]!=0 and m[i,j+1] % m[i,j+2] == 0:
                pieces.append(m[i,j+1]//m[i,j+2])
            if m[i,j+1]!=0 and m[i,j+2] % m[i,j+1] == 0:
                pieces.append(m[i,j+2]//m[i,j+1])
                
        if [i,j-1] in visited and [i,j-2] in visited:
            pieces.append(m[i,j-1]+m[i,j-2])
            pieces.append(abs(m[i,j-1]-m[i,j-2]))
            pieces.append(m[i,j-1]*m[i,j-2])
            if m[i,j-2]!=0 and m[i,j-1] % m[i,j-2] == 0:
                pieces.append(m[i,j-1]//m[i,j-2])
            if m[i,j-1]!=0 and m[i,j-2] % m[i,j-1] == 0:
                pieces.append(m[i,j-2]//m[i,j-1])   
                
        if [i+1,j] in visited and [i+2,j] in visited:
            pieces.append(m[i+1,j]+m[i+2,j])
            pieces.append(abs(m[i+1,j]-m[i+2,j]))
            pieces.append(m[i+1,j]*m[i+2,j])
            if m[i+2,j]!=0 and m[i+1,j] % m[i+2,j] == 0:
                pieces.append(m[i+1,j]//m[i+2,j])
            if m[i+1,j]!=0 and m[i+2,j] % m[i+1,j] == 0:
                pieces.append(m[i+2,j]//m[i+1,j])
                
    if m[i,j]=='+':
        if [i-1,j] in visited and [i-2,j] in visited:
            pieces.append(m[i-1,j]+m[i-2,j])
                
        if [i,j+1] in visited and [i,j+2] in visited:
            pieces.append(m[i,j+1]+m[i,j+2])
                
        if [i,j-1] in visited and [i,j-2] in visited:
            pieces.append(m[i,j-1]+m[i,j-2])
                
        if [i+1,j] in visited and [i+2,j] in visited:
            pieces.append(m[i+1,j]+m[i+2,j])
            
    if m[i,j]=='-':
        if [i-1,j] in visited and [i-2,j] in visited:
            pieces.append(abs(m[i-1,j]-m[i-2,j]))
                
        if [i,j+1] in visited and [i,j+2] in visited:
            pieces.append(abs(m[i,j+1]-m[i,j+2]))
                
        if [i,j-1] in visited and [i,j-2] in visited:
            pieces.append(abs(m[i,j-1]-m[i,j-2]))
                
        if [i+1,j] in visited and [i+2,j] in visited:
            pieces.append(abs(m[i+1,j]-m[i+2,j]))
            
    if m[i,j]=='*':
        if [i-1,j] in visited and [i-2,j] in visited:
            pieces.append(m[i-1,j]*m[i-2,j])
                
        if [i,j+1] in visited and [i,j+2] in visited:
            pieces.append(m[i,j+1]*m[i,j+2])
                
        if [i,j-1] in visited and [i,j-2] in visited:
            pieces.append(m[i,j-1]*m[i,j-2])  
                
        if [i+1,j] in visited and [i+2,j] in visited:
            pieces.append(m[i+1,j]*m[i+2,j])
            
    if m[i,j]=='/':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i-2,j]!=0 and m[i-1,j] % m[i-2,j] == 0:
                pieces.append(m[i-1,j]//m[i-2,j])
            if m[i-1,j]!=0 and m[i-2,j] % m[i-1,j] == 0:
                pieces.append(m[i-2,j]//m[i-1,j])
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j+2]!=0 and m[i,j+1] % m[i,j+2] == 0:
                pieces.append(m[i,j+1]//m[i,j+2])
            if m[i,j+1]!=0 and m[i,j+2] % m[i,j+1] == 0:
                pieces.append(m[i,j+2]//m[i,j+1])
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j-2]!=0 and m[i,j-1] % m[i,j-2] == 0:
                pieces.append(m[i,j-1]//m[i,j-2])
            if m[i,j-1]!=0 and m[i,j-2] % m[i,j-1] == 0:
                pieces.append(m[i,j-2]//m[i,j-1])   
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i+2,j]!=0 and m[i+1,j] % m[i+2,j] == 0:
                pieces.append(m[i+1,j]//m[i+2,j])
            if m[i+1,j]!=0 and m[i+2,j] % m[i+1,j] == 0:
                pieces.append(m[i+2,j]//m[i+1,j])
    
    return pieces
        
#Function to calculate the score of a given piece
def calculate_score(i,j):
    score=0
    if m_original[i,j]==0:
        
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i-1,j]+m[i-2,j]==m[i,j] or abs(m[i-1,j]-m[i-2,j])==m[i,j] or m[i-1,j]*m[i-2,j]==m[i,j] or (m[i-1,j]!=0 and m[i-2,j]/m[i-1,j]==m[i,j]) or (m[i-2,j]!=0 and m[i-1,j]//m[i-2,j]==m[i,j]):
                score=score+m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j+1]+m[i,j+2]==m[i,j] or abs(m[i,j+1]-m[i,j+2])==m[i,j] or m[i,j+1]*m[i,j+2]==m[i,j] or (m[i,j+1]!=0 and m[i,j+2]/m[i,j+1]==m[i,j]) or (m[i,j+2]!=0 and m[i,j+1]//m[i,j+2]==m[i,j]):
                score=score+m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j-1]+m[i,j-2]==m[i,j] or abs(m[i,j-1]-m[i,j-2])==m[i,j] or m[i,j-1]*m[i,j-2]==m[i,j] or (m[i,j-1]!=0 and m[i,j-2]/m[i,j-1]==m[i,j]) or (m[i,j-2]!=0 and m[i,j-1]//m[i,j-2]==m[i,j]):
                score=score+m[i,j]
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i+1,j]+m[i+2,j]==m[i,j] or abs(m[i+1,j]-m[i+2,j])==m[i,j] or m[i+1,j]*m[i+2,j]==m[i,j] or (m[i+1,j]!=0 and m[i+2,j]/m[i+1,j]==m[i,j]) or (m[i+2,j]!=0 and m[i+1,j]//m[i+2,j]==m[i,j]):
                score=score+m[i,j]
                
    if m_original[i,j]=='2x':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i-1,j]+m[i-2,j]==m[i,j] or abs(m[i-1,j]-m[i-2,j])==m[i,j] or m[i-1,j]*m[i-2,j]==m[i,j] or (m[i-1,j]!=0 and m[i-2,j]/m[i-1,j]==m[i,j]) or (m[i-2,j]!=0 and m[i-1,j]//m[i-2,j]==m[i,j]):
                score=score+2*m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j+1]+m[i,j+2]==m[i,j] or abs(m[i,j+1]-m[i,j+2])==m[i,j] or m[i,j+1]*m[i,j+2]==m[i,j] or (m[i,j+1]!=0 and m[i,j+2]/m[i,j+1]==m[i,j]) or (m[i,j+2]!=0 and m[i,j+1]//m[i,j+2]==m[i,j]):
                score=score+2*m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j-1]+m[i,j-2]==m[i,j] or abs(m[i,j-1]-m[i,j-2])==m[i,j] or m[i,j-1]*m[i,j-2]==m[i,j] or (m[i,j-1]!=0 and m[i,j-2]/m[i,j-1]==m[i,j]) or (m[i,j-2]!=0 and m[i,j-1]//m[i,j-2]==m[i,j]):
                score=score+2*m[i,j]
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i+1,j]+m[i+2,j]==m[i,j] or abs(m[i+1,j]-m[i+2,j])==m[i,j] or m[i+1,j]*m[i+2,j]==m[i,j] or (m[i+1,j]!=0 and m[i+2,j]/m[i+1,j]==m[i,j]) or (m[i+2,j]!=0 and m[i+1,j]//m[i+2,j]==m[i,j]):
                score=score+2*m[i,j]
                
    if m_original[i,j]=='3x':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i-1,j]+m[i-2,j]==m[i,j] or abs(m[i-1,j]-m[i-2,j])==m[i,j] or m[i-1,j]*m[i-2,j]==m[i,j] or (m[i-1,j]!=0 and m[i-2,j]/m[i-1,j]==m[i,j]) or (m[i-2,j]!=0 and m[i-1,j]//m[i-2,j]==m[i,j]):
                score=score+3*m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j+1]+m[i,j+2]==m[i,j] or abs(m[i,j+1]-m[i,j+2])==m[i,j] or m[i,j+1]*m[i,j+2]==m[i,j] or (m[i,j+1]!=0 and m[i,j+2]/m[i,j+1]==m[i,j]) or (m[i,j+2]!=0 and m[i,j+1]//m[i,j+2]==m[i,j]):
                score=score+3*m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j-1]+m[i,j-2]==m[i,j] or abs(m[i,j-1]-m[i,j-2])==m[i,j] or m[i,j-1]*m[i,j-2]==m[i,j] or (m[i,j-1]!=0 and m[i,j-2]/m[i,j-1]==m[i,j]) or (m[i,j-2]!=0 and m[i,j-1]//m[i,j-2]==m[i,j]):
                score=score+3*m[i,j]
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i+1,j]+m[i+2,j]==m[i,j] or abs(m[i+1,j]-m[i+2,j])==m[i,j] or m[i+1,j]*m[i+2,j]==m[i,j] or (m[i+1,j]!=0 and m[i+2,j]/m[i+1,j]==m[i,j]) or (m[i+2,j]!=0 and m[i+1,j]//m[i+2,j]==m[i,j]):
                score=score+3*m[i,j]
                
    if m_original[i,j]=='+':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i,j]==m[i-1,j]+m[i-2,j]:
                score=score+m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j]==m[i,j+1]+m[i,j+2]:
                score=score+m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j]==m[i,j-1]+m[i,j-2]:
                score=score+m[i,j]
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i,j]==m[i+1,j]+m[i+2,j]:
                score=score+m[i,j]
            
    if m_original[i,j]=='-':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i,j]==abs(m[i-1,j]-m[i-2,j]):
                score=score+m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j]==abs(m[i,j+1]-m[i,j+2]):
                score=score+m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j]==abs(m[i,j-1]-m[i,j-2]):
                score=score+m[i,j]
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i,j]==abs(m[i+1,j]-m[i+2,j]):
                score=score+m[i,j]
            
    if m_original[i,j]=='*':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i,j]==m[i-1,j]*m[i-2,j]:
                score=score+m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j]==m[i,j+1]*m[i,j+2]:
                score=score+m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j]==m[i,j-1]*m[i,j-2]:
                score=score+m[i,j]
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i,j]==m[i+1,j]*m[i+2,j]:
                score=score+m[i,j]
            
    if m_original[i,j]=='/':
        if [i-1,j] in visited and [i-2,j] in visited:
            if m[i-2,j]!=0 and m[i-1,j] // m[i-2,j] == m[i,j]:
                score=score+m[i,j]
            if m[i-1,j]!=0 and m[i-2,j] // m[i-1,j] == m[i,j]:
                score=score+m[i,j]
                
        if [i,j+1] in visited and [i,j+2] in visited:
            if m[i,j+2]!=0 and m[i,j+1] // m[i,j+2] == m[i,j]:
                score=score+m[i,j]
            if m[i,j+1]!=0 and m[i,j+2] // m[i,j+1] == m[i,j]:
                score=score+m[i,j]
                
        if [i,j-1] in visited and [i,j-2] in visited:
            if m[i,j-2]!=0 and m[i,j-1] // m[i,j-2] == m[i,j]:
                score=score+m[i,j]
            if m[i,j-1]!=0 and m[i,j-2] // m[i,j-1] == m[i,j]:
                score=score+m[i,j]  
                
        if [i+1,j] in visited and [i+2,j] in visited:
            if m[i+2,j]!=0 and m[i+1,j] // m[i+2,j] == m[i,j]:
                score=score+m[i,j]
            if m[i+1,j]!=0 and m[i+2,j] // m[i+1,j] == m[i,j]:
                score=score+m[i,j]
    
    return score

#Function that return the difference between game boards
def board_difference(image1,image2,threshold_value=85):
    diff=cv.absdiff(image1,image2)

    diff=cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    
    _,binary_diff=cv.threshold(diff,threshold_value,255,cv.THRESH_BINARY)
    
    kernel=np.ones((5,5),np.uint8)
    
    binary_diff=cv.erode(binary_diff,kernel)

    binary_diff_bgr=cv.cvtColor(binary_diff,cv.COLOR_GRAY2BGR)
    
    result=binary_diff_bgr
    
    return result

#Function that return the pozition of the new piece
def find_tile_position(thresh,lines_horizontal,lines_vertical):
    maxi=0
    index=[]
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0] +10 
            y_max = lines_vertical[j + 1][1][0] -10
            x_min = lines_horizontal[i][0][1] +10
            x_max = lines_horizontal[i + 1][1][1] -10
            patch = thresh[x_min:x_max, y_min:y_max].copy()

            if [(x_min-10)//100,(y_min-10)//100] in possible_moves and [(x_min-10)//100,(y_min-10)//100] not in visited:
                medie=np.mean(patch) 
                if medie>maxi:
                    maxi=medie
                    index.append([i,j])    
    return index[-1]

#Function that returs the number on a tile
def what_number(image,patch,lines_horizontal,lines_vertical):
    x_min=(patch[0])*100+10
    x_max=x_min+90
    y_min=patch[1]*100+10
    y_max=y_min+90
    
    cut=image[x_min:x_max,y_min:y_max].copy()
    nr_contours=count_contours(cut)
    cut=cv.cvtColor(cut,cv.COLOR_BGR2GRAY)
    
    cut=image[x_min-10:x_max+10,y_min-10:y_max+10].copy()
    cut=cv.cvtColor(cut,cv.COLOR_BGR2GRAY)
    
    possible_pieces=calculate_possible_pieces(patch[0],patch[1])
    #print('Possible pieces:',possible_pieces)
    
    maxi=-np.inf
    
    for j in pieces:
        if j in possible_pieces and len(str(j))==nr_contours:
            img_template=cv.imread('Templates/'+str(j)+'.jpg')
            img_template=cv.cvtColor(img_template,cv.COLOR_BGR2GRAY)
            
            corr=cv.matchTemplate(cut,img_template,cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)
            if corr>maxi:
                
                maxi=corr
                cif=j
    #Uncomment to see the 
    #show_image(f'Guess:{cif,maxi}',cut)
    m[patch[0],patch[1]]=cif
    return cif

#Function to find the first player in a game and the number of turns each player takes
def give_player_turns(game_nr,input_folder):
    txt_file=os.path.join(input_folder,f'{game_nr}_turns.txt')
    if not os.path.exists(txt_file):
        return None,[]
    player_start=[]
    with open(txt_file,'r') as f:
        player_list=f.read().split()
        current_player=player_list[0]
        for i,obj in enumerate(player_list):
            if obj=='Player1' or obj=='Player2':
                player_start.append(int(player_list[i+1]))
    player_turns=[]
    for i in range(len(player_start)):
        if i<len(player_start)-1:
            player_turns.append(abs(player_start[i]-player_start[i+1]))
        else:
            player_turns.append(abs(player_start[i]-50)+1)
    return current_player,player_turns

m=np.array([['3x',0,0,0,0,0,'3x','3x',0,0,0,0,0,'3x'],
                            [0,'2x',0,0,'/',0,0,0,0,'/',0,0,'2x',0],
                            [0,0,'2x',0,0,'-',0,0,'-',0,0,'2x',0,0],
                            [0,0,0,'2x',0,0,'+','*',0,0,'2x',0,0,0],
                            [0,'/',0,0,'2x',0,'*','+',0,'2x',0,0,'/',0],
                            [0,0,'-',0,0,0,0,0,0,0,0,'-',0,0],
                            ['3x',0,0,'*','+',0,1,2,0,'*','+',0,0,'3x'],
                            ['3x',0,0,'+','*',0,3,4,0,'+','*',0,0,'3x'],
                            [0,0,'-',0,0,0,0,0,0,0,0,'-',0,0],
                            [0,'/',0,0,'2x',0,'+','*',0,'2x',0,0,'/',0],
                            [0,0,0,'2x',0,0,'*','+',0,0,'2x',0,0,0],
                            [0,0,'2x',0,0,'-',0,0,'-',0,0,'2x',0,0],
                            [0,'2x',0,0,'/',0,0,0,0,'/',0,0,'2x',0],
                            ['3x',0,0,0,0,0,'3x','3x',0,0,0,0,0,'3x']],dtype=object)

m_original=np.array([['3x',0,0,0,0,0,'3x','3x',0,0,0,0,0,'3x'],
                            [0,'2x',0,0,'/',0,0,0,0,'/',0,0,'2x',0],
                            [0,0,'2x',0,0,'-',0,0,'-',0,0,'2x',0,0],
                            [0,0,0,'2x',0,0,'+','*',0,0,'2x',0,0,0],
                            [0,'/',0,0,'2x',0,'*','+',0,'2x',0,0,'/',0],
                            [0,0,'-',0,0,0,0,0,0,0,0,'-',0,0],
                            ['3x',0,0,'*','+',0,1,2,0,'*','+',0,0,'3x'],
                            ['3x',0,0,'+','*',0,3,4,0,'+','*',0,0,'3x'],
                            [0,0,'-',0,0,0,0,0,0,0,0,'-',0,0],
                            [0,'/',0,0,'2x',0,'+','*',0,'2x',0,0,'/',0],
                            [0,0,0,'2x',0,0,'*','+',0,0,'2x',0,0,0],
                            [0,0,'2x',0,0,'-',0,0,'-',0,0,'2x',0,0],
                            [0,'2x',0,0,'/',0,0,0,0,'/',0,0,'2x',0],
                            ['3x',0,0,0,0,0,'3x','3x',0,0,0,0,0,'3x']],dtype=object)
        
pieces=[0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,11,12,13,14,15,16,17,18,19,20,21,24,25,27,28,30,32,35,36,40,42,45,48,49,50,54,56,60,63,64,70,72,80,81,90]

possible_moves=[[5,6],[5,7],[6,5],[6,8],[7,5],[7,8],[8,6],[8,7]]

visited=[[6,6],[6,7],[7,6],[7,7]]

lines_horizontal=[]
for i in range(0,1401,100):
    l=[]
    l.append((0,i))
    l.append((1399,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,1401,100):
    l=[]
    l.append((i,0))
    l.append((i,1399))
    lines_vertical.append(l)
    
top_left_crop = (850, 1750)
top_right_crop = (2340, 1750)
bottom_left_crop = (850, 3260)
bottom_right_crop = (2340, 3260)
    
######################################################################
#Change the input and output folder if necesarry
input_folder='antrenare/'
output_folder='311_Oproiu_Matei/'
######################################################################

os.makedirs(output_folder, exist_ok=True)
files=sorted(os.listdir(input_folder))
prev_img=cv.imread('imagini_auxiliare_crop/cropped_02.jpg')
game_index = 1
move_counter=0
nr_correct_poz=0
nr_correct_dig=0
lista_proaste=[]
nr_ghiceli=0
current_player,player_turns=give_player_turns(game_index,input_folder)
turn_index=1
player1_score=0
player2_score=0
game_scores=[]
for file in files:
    if file[-3:]=='jpg':
        
        print(f'Checking gameboard nr: {file[:-4]}')
    
        img = cv.imread(input_folder+file)
        
        cropped_img=img[top_left_crop[1]:bottom_left_crop[1], top_left_crop[0]:top_right_crop[0]]
        
        top_left,bottom_right=extrage_careu(cropped_img)

        img_flip=cv.flip(cropped_img,1)

        height, width = cropped_img.shape[:2]

        top_right,bottom_left=extrage_careu(img_flip)
        top_right=(width-top_right[0],top_right[1])
        bottom_left=(width-bottom_left[0],bottom_left[1])

        game_board=transform(cropped_img,top_left,top_right,bottom_left,bottom_right)
        
        #show_image(f'{file}',game_board)
        res=board_difference(prev_img,game_board)
        
        matrice=find_tile_position(res,lines_horizontal,lines_vertical)
        predicted_position=f"{matrice[0]+1}{chr(64 + matrice[1]+1)}"
        
        predicted_number=what_number(game_board,matrice,lines_horizontal,lines_vertical)
        predicted_number=str(predicted_number)
        
        
        result=os.path.join(output_folder, f"{file[:-4]}.txt")
        with open(result,'w') as file:
            file.write(f"{predicted_position} {predicted_number}")
        
        
        visited.append(matrice)
        possible_moves.append([matrice[0],matrice[1]-1])
        possible_moves.append([matrice[0],matrice[1]+1])
        possible_moves.append([matrice[0]-1,matrice[1]])
        possible_moves.append([matrice[0]+1,matrice[1]])
        prev_img=game_board
        
        if player_turns and turn_index<player_turns[0]:
            score=calculate_score(matrice[0],matrice[1])
            turn_index+=1
            if current_player=='Player1':
                player1_score+=score
            else:
                player2_score+=score
        else:
            #calculate_score
            score=calculate_score(matrice[0],matrice[1])
            if current_player=='Player1':
                player1_score+=score
                game_scores.append(player1_score)
                player1_score=0
            elif current_player=='Player2':
                player2_score+=score
                game_scores.append(player2_score)
                player2_score=0
                
            turn_index=1
            player_turns=player_turns[1:]
            if current_player=='Player1':
                current_player='Player2'
            elif current_player=='Player2':
                current_player='Player1'
                   
        move_counter=move_counter+1
        if move_counter==50:
            move_counter=0
            input_file = os.path.join(input_folder, f"{game_index}_turns.txt")
            output_file = os.path.join(output_folder, f"{game_index}_scores.txt")
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                lines = infile.readlines()
                line_count = len(lines)
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) == 2 and i < len(game_scores):  
                        player = parts[0]
                        round_number = parts[1]
                        score = game_scores[i]
                        if i==line_count-1:
                            updated_line = f"{player} {round_number} {score}"
                        else:
                            updated_line = f"{player} {round_number} {score}\n"
                        
                        outfile.write(updated_line)
            game_scores=[]
            
            game_index+=1
            current_player,player_turns=give_player_turns(game_index,input_folder)
            
            prev_img=cv.imread('imagini_auxiliare_crop/cropped_02.jpg')
            
            possible_moves=[[5,6],[5,7],[6,5],[6,8],[7,5],[7,8],[8,6],[8,7]]
            visited=[[6,6],[6,7],[7,6],[7,7]]
            m=np.array([['3x',0,0,0,0,0,'3x','3x',0,0,0,0,0,'3x'],
                                        [0,'2x',0,0,'/',0,0,0,0,'/',0,0,'2x',0],
                                        [0,0,'2x',0,0,'-',0,0,'-',0,0,'2x',0,0],
                                        [0,0,0,'2x',0,0,'+','*',0,0,'2x',0,0,0],
                                        [0,'/',0,0,'2x',0,'*','+',0,'2x',0,0,'/',0],
                                        [0,0,'-',0,0,0,0,0,0,0,0,'-',0,0],
                                        ['3x',0,0,'*','+',0,1,2,0,'*','+',0,0,'3x'],
                                        ['3x',0,0,'+','*',0,3,4,0,'+','*',0,0,'3x'],
                                        [0,0,'-',0,0,0,0,0,0,0,0,'-',0,0],
                                        [0,'/',0,0,'2x',0,'+','*',0,'2x',0,0,'/',0],
                                        [0,0,0,'2x',0,0,'*','+',0,0,'2x',0,0,0],
                                        [0,0,'2x',0,0,'-',0,0,'-',0,0,'2x',0,0],
                                        [0,'2x',0,0,'/',0,0,0,0,'/',0,0,'2x',0],
                                        ['3x',0,0,0,0,0,'3x','3x',0,0,0,0,0,'3x']],dtype=object)
            
print('Program finished')
print(f'The results are in the {output_folder} folder')
