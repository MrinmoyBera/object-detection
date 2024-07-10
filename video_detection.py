#import some libraries
import cv2
import numpy as np


#class label
coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , 
 "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" ,
"stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , 
"sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" ,
 "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]




#for detection function
video_path =  "C:\\Users\Mrinmoy Bera\Downloads\\test_video.mp4"
weightsPath = "C:\\Users\\Mrinmoy Bera\\Downloads\\weight.pb"
configPath = "C:\\Users\\Mrinmoy Bera\\Downloads\\config.pbtxt"

def object_detector(weightsPath, configPath, video_path) :
  # load the pretrained model
  net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

  cap = cv2.VideoCapture(video_path)
  while True :
      ret, frame = cap.read()

      # H contains the height of the image and W contains width of the image
      (H, W) = frame.shape[:2]
    
      # Show the given image
      cv2.imshow('Givenvideo', frame)
      
      # cv2.dnn.blobFromImage is a function that transforms an image into a blob, which is a batch of images.
      # This transformation is necessary because deep learning models typically expect input in a specific format.
      blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    
      # Set the input to the network
      net.setInput(blob)
    
      # Perform forward pass to get the output of the detection layer
      boxes = net.forward(["detection_out_final"])
    
      # Process the output
      # output has 7 columns 0.batch_id, 1.class_id, 2.confidence, 3. X1, 4.Y1, 5. X2, 6. Y2
      output = boxes[0].squeeze()
    
      # Take only those box which have confidence is greater than 0.8(ie. 80%).
      num = np.argwhere(output[: , 2] > 0.8).shape[0]
      font = cv2.FONT_HERSHEY_SIMPLEX
      img = frame
      for i in range(num):
        x1n , y1n , x2n , y2n = output[i , 3:]
        x1 = int(x1n * W)
        y1 = int(y1n * H)
        x2 = int(x2n * W)
        y2 = int(y2n * H)
        img = cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (0 , 255 , 0) , 3)
        class_name = coco_names[int(output[i , 1])]
        img = cv2.putText(img , class_name , (x1 , y1 - 10) , font , 0.5 , 
                          (255 , 0 , 0) , 1 , cv2.LINE_AA)
      #showing image with object detection
      img = cv2.resize(img, (1000,550))
      cv2.imshow('After_detection', img)
      # Wait for a key event (waits indefinitely for any key to be pressed)
      k=cv2.waitKey(10)
      if k == ord('q') :
          break




# Close all OpenCV windows
cv2.destroyAllWindows()
object_detector(weightsPath, configPath, video_path)