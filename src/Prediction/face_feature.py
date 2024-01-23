import math

# Usage: Initialize a Feature instasnce with 68 points landmark array
#		 then access ft.feature to get the 5 features

class Feature:
    points = [] # original 68 key points model
    feature = [None] * 5 # f0-f4

    def __init__(self, points_68):
        if len(points_68) != 68:
            print("Not 68 landmarks")
            exit()

        self.points = [p for p in points_68] # points to list of point

        self.feature[0] = abs(90 - abs(Feature.angle(self.points[30], self.points[27]))) # straightness of nose
        self.feature[1] = abs(180 - abs(Feature.angle(self.points[48], self.points[54]))) # angle between 2 outer mouth points
        
        left_eye_average_height = (Feature.dist(self.points[37], self.points[41]) + Feature.dist(self.points[38], self.points[40])) / 2
        right_eye_average_height = (Feature.dist(self.points[43], self.points[47]) + Feature.dist(self.points[44], self.points[46])) / 2
        self.feature[2] = left_eye_average_height - right_eye_average_height # different between average eyes height
        
        self.feature[3] = Feature.angle(self.points[26], self.points[17]) # angle between outermost points of eyebrows

        left_lips_segment = Feature.segment([*self.points[48:51+1], *self.points[57:59+1]])
        right_lips_segment = Feature.segment(self.points[51:57+1])
        self.feature[4] = left_lips_segment - right_lips_segment
        
    # Given point a and b return their euclidean distance
    @staticmethod
    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    
    # Given point a and b return their slope
    @staticmethod
    def slope(a, b):
        return abs((a.y - b.y) / (a.x - b.x))
    
    # Given point a and b return angle of delta x and y in degree
    @staticmethod
    def angle(a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        return math.atan2(dy, dx) * 180 / math.pi
    
    # Given point in order, 
    @staticmethod
    def segment(points):
        sum = 0
        for i in range(len(points) - 1):
            sum += Feature.dist(points[i], points[i + 1])
        sum += Feature.dist(points[0], points[-1])
        return sum