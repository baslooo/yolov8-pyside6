from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# model.predictor()
# results = model(source='cxk.mp4',stream=True) # predict on an image
results = model(source='cxk.mp4',stream=False)  # predict on an image
# model.predictor.show('cxk.mp4)
# results.model.show()
# results.predictor.show()
# while True:
#     results.predictor.show()
# for result in results:
#     print(result)

count =0
while True:
    if 条件1:
        一些代码
    count+=1
    if count:
        一些代码
上述代码如果不进入条件一的代码就能正常运行,反之报错 no attribute 'count',为什么