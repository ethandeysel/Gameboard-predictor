import cv2
import easyocr
import numpy as np

# Initialize once when the module is imported
reader = easyocr.Reader(['en'], gpu=False)

def number_extractor(image):
    gray_tile = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    padding = 15
    padded_tile = cv2.copyMakeBorder(
        thresh, padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    upscaled = cv2.resize(padded_tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    results = reader.readtext(upscaled)
    if not results:
        return None 
    text = results[0][1]
    clean = ''.join(filter(str.isdigit, text))
    return int(clean) if clean else None

def process_all_boards(total_weeks=19, image_dir='data/images'):
    imgs = []
    for k in range(total_weeks):
        path = f'{image_dir}/Image{k+1}.jpg' 
        img = cv2.imread(path)
        
        if img is None:
            print(f"Warning: Could not read image at {path}. Skipping...")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img_rgb.shape
        rows, cols = 5, 4
        row_step, col_step = height / rows, width / cols
        
        vals = [] 
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * row_step, (r + 1) * row_step
                x1, x2 = c * col_step, (c + 1) * col_step
        
                tile = img_rgb[int(y1):int(y2), int(x1):int(x2)]
                t_h, t_w, _ = tile.shape
                number_img = tile[int(t_h * 0.5):t_h, 0:t_w]
                
                value = number_extractor(number_img)
                vals.append(value)
                
        imgs.append(vals) 
        print(f"Processed Week {k+1}")
        
    return imgs