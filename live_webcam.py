import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import ObjectClassification

classes = ['cutlery', 'hair_brushes', 'keys', 'medications', 'specs']
# {'cutlery': 0, 'hair_brushes': 1, 'keys': 2, 'medications': 3, 'specs': 4}
model = ObjectClassification()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break 

    # opencv uses BGR, convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #convert to PIL image
    image = Image.fromarray(rgb)

    #preprocess
    input_tensor = transform(image).unsqueeze(0)


    #inference
    with torch.no_grad():
        output = model(input_tensor)
        # prediction = torch.argmax(output, dim=1).item()

        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

        confidence = confidence.item() * 100
        prediction = prediction.item()         

    #display prediction
    if confidence > 70:
        cv2.putText(frame,
                    f"Class: {classes[prediction]} \n Confidence Score:{confidence}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2)
    else:
        cv2.putText(frame,
                    f"Class: Unknown Object \n possibility: {classes[prediction]} ({confidence}%)",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2)

    cv2.imshow("Object Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()