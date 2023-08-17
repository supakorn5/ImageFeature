from fastapi import FastAPI, HTTPException, Request
import cv2
import numpy as np
import base64

app = FastAPI()


def hog(base64_img):
    img_new = cv2.resize(base64_img, (128, 128), cv2.INTER_AREA)
    win_size = img_new.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    # Set the parameters of the HOG descriptor using the variables defined above
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, num_bins)

    # Compute the HOG Descriptor for the grayscale image
    hog_descriptor = hog.compute(img_new)
    return hog_descriptor


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/gethog")
async def gethog(request: Request):
    try:
        item = await request.json()
        item_str = item['img']
        img = readb64(item_str)
        hog_descriptor = hog(img)
        return {"Hog": hog_descriptor.tolist()}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
