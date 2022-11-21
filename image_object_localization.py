import cv2
from imread_from_url import imread_from_url

from oln import ObjectLocalizationNet, remove_initializer_from_input

# Initialize object localizer
model_path = "models/oln_720x1280.onnx"
print('removing initializer')
remove_initializer_from_input(model_path, model_path) # Remove unused nodes
print('initializing network')
localizer = ObjectLocalizationNet(model_path, threshold=0.7)


print('reading image')
img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/2/2b/Interior_design_865875.jpg")

print('running inference')
# Update object localizer
detections, scores = localizer(img)

print(detections)
print(scores)

combined_img = localizer.draw_detections(img)
# cv2.namedWindow("Objects", cv2.WINDOW_NORMAL)
# cv2.imwrite("output1.png", combined_img)
# cv2.waitKey(0)

cv2.imwrite("test_output/output.jpg", combined_img)