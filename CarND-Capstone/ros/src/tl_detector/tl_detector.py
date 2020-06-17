#!/usr/bin/env python
import rospy, rospkg, os
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.image_counter = 0

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        rospack = rospkg.RosPack()
        proj_path = rospack.get_path('tl_detector')
        self.model_path = os.path.join(proj_path, 'light_classification/model')
        print('model_path: {}'.format(self.model_path))
        
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.image_boxes_pub = rospy.Publisher('/image_boxes', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.model_path)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_2d = None
        self.waypoints_tree = None
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if self.waypoints_2d == None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            print('waypoints_tree loaded')
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
#         print('self.lights :{}'.format(self.lights))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.image_counter < 3:
            self.image_counter += 1
            return
        self.image_counter = 0
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = -1
        if self.waypoints_tree:
            x = pose.position.x
            y = pose.position.y
            closest_idx = self.waypoints_tree.query([x,y],1)[1]
        
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        state, result_image = self.light_classifier.get_classification(cv_image)
        print('predicted state: {}'.format(state))
        image_msg = self.bridge.cv2_to_imgmsg(result_image, "rgb8")
        return state, image_msg

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.waypoints_tree:
            car_position_idx = self.get_closest_waypoint(self.pose.pose)
        #TODO find the closest visible traffic light (if one exists)
#         if self.waypoints and self.waypoints_tree and self.lights:
            
#             min_dist = len(self.waypoints.waypoints)
            min_dist = 200
#             print('min_dist: {}'.format(min_dist))
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line_pose = Pose()
                line =  stop_line_positions[i]
                line_pose.position.x = line[0]
                line_pose.position.y = line[1]
                temp_wp_idx = self.get_closest_waypoint(line_pose)

                # Loop to find the closest light
                dist = temp_wp_idx - car_position_idx
                if dist >= 0 and dist < min_dist:
                    closest_light = light
                    min_dist = dist
                    light_wp_idx = temp_wp_idx

        if closest_light:
            state, image_msg = self.get_light_state(light)
            print('true state: {}'.format(light.state))
            self.image_boxes_pub.publish(image_msg)
            return light_wp_idx, state
        
        self.image_boxes_pub.publish(self.camera_image)
            
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
