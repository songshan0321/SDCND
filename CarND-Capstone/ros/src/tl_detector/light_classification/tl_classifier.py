from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf
import cv2
import os

# from object_detection.utils import label_map_util
# import visualization_utils as vis_util

class TLClassifier(object):
    def __init__(self, model_dir):
        
        self.class_map = {
            1: TrafficLight.GREEN,
            2: TrafficLight.YELLOW,
            3: TrafficLight.RED
        }
        
        self.current_light = TrafficLight.UNKNOWN
        model_file = 'ssd_mobilenet/frozen_inference_graph.pb'
        model_path = os.path.join(model_dir, model_file)

        self.detection_graph = self.load_model(model_path)
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
#         image_np = np.asarray(image)
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                
                image_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num) = sess.run([detect_boxes, detect_scores, detect_classes, num_detections],
                                                        feed_dict={image_tensor: image_expanded})
                
                self.draw_boxes(image, 
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),                  
                                np.squeeze(scores),
                                0.5)

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
            
#                 print(boxes)
#                 print(scores)
#                 print(classes)
                
                keep = scores > 0.5
                if np.any(keep):

                    members, index, counts = np.unique(classes, return_inverse=True, return_counts=True)
                    member_scores = np.zeros((len(members),))

                    for i in range(len(members)):
                        member_scores[i] = np.sum(scores[index == i])

                        select = np.argmax(member_scores)
                        winner = members[select]

                        state = self.class_map[winner]

                else:
                    state = TrafficLight.UNKNOWN

        return state, image
    
    def load_model(self, file_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph
    
    def draw_boxes(self, image, boxes, classes, scores, min_score):
        for i in range(len(scores)):
            score = scores[i]
            if score > min_score:
                box = boxes[i]
                start_point = (int(box[1]*image.shape[1]), int(box[0]*image.shape[0]))
                end_point = (int(box[3]*image.shape[1]), int(box[2]*image.shape[0]))
                if classes[i] == 1:
                    color = (0, 255, 0)
                elif classes[i] == 2:
                    color = (255, 255, 0)
                elif classes[i] == 3:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                    
                thickness = 5
                
                image = cv2.rectangle(image, start_point, end_point, color, thickness) 
   
