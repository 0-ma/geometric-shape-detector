import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification 
import requests
def main():
    labels =  [
        "None",
        "Circle",
        "Triangle",
        "Square",
        "Pentagon",
        "Hexagon"
    ] 
        
    #Local file equivalent
    # images = [Image.open("input/exemple_circle.jpg"), 
    #           Image.open("input/exemple_pentagone.jpg")]
    
    images = [Image.open(requests.get("https://raw.githubusercontent.com/0-ma/geometric-shape-detector/main/input/exemple_circle.jpg", stream=True).raw), 
            Image.open(requests.get("https://raw.githubusercontent.com/0-ma/geometric-shape-detector/main/input/exemple_pentagone.jpg", stream=True).raw)]
    feature_extractor = AutoImageProcessor.from_pretrained('0-ma/vit-geometric-shapes-tiny')
    model = AutoModelForImageClassification.from_pretrained('0-ma/vit-geometric-shapes-tiny')
    inputs = feature_extractor(images=images, return_tensors="pt")
    logits = model(**inputs)['logits'].cpu().detach().numpy()
    predictions = np.argmax(logits, axis=1)    
    predicted_labels = [labels[prediction] for prediction in predictions]
    print(predicted_labels)





        

if __name__ == "__main__":
    main()