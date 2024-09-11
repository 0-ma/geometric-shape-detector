import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification 

def main():
    labels =  [
        "Only text",
        "Circle",
        "Triangle",
        "Square",
        "Pentagon",
        "Hexagon"
    ] 
        
    images = [Image.open("input/exemple_circle.jpg"), 
              Image.open("input/exemple_pentagone.jpg")]
    feature_extractor = AutoImageProcessor.from_pretrained('0-ma/vit-geometric-shapes-tiny')
    model = AutoModelForImageClassification.from_pretrained('0-ma/vit-geometric-shapes-tiny')
    inputs = feature_extractor(images=images, return_tensors="pt")
    logits = model(**inputs)['logits'].cpu().detach().numpy()
    predictions = np.argmax(logits, axis=1)    
    predicted_labels = [labels[prediction] for prediction in predictions]
    print(predicted_labels)





        

if __name__ == "__main__":
    main()