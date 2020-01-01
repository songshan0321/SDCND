import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Lane():
    
    def __init__(self, lane_detected=False):
        self.lane_detected = lane_detected
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        self.left_fit_trial = None
        self.right_fit_trial = None
    
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 100

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img
    
    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        ### Fit a second order polynomial to each with np.polyfit() ###
        self.left_fit_trial = np.polyfit(lefty, leftx, 2)
        self.right_fit_trial = np.polyfit(righty, rightx, 2)
        
        left_curverad, right_curverad = self.measure_curvature_real_input(self.left_fit_trial, self.right_fit_trial, self.ploty)
        if self.check_lane_validity(left_curverad, right_curverad):
            self.left_fit = self.left_fit_trial
            self.right_fit = self.right_fit_trial
            self.lane_detected = True
        else:
            self.lane_detected = False
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        return left_fitx, right_fitx, ploty


    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        left_points = (np.asarray([left_fitx, ploty]).T).astype(np.int32)
        right_points = (np.asarray([right_fitx, ploty]).T).astype(np.int32)
        cv2.polylines(result, [left_points], False, (255,255,255), thickness=2)
        cv2.polylines(result, [right_points], False, (255,255,255), thickness=2)
        ## End visualization steps ##

        return result


    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            
        if self.lane_detected:
            out_img = self.search_around_poly(binary_warped)

        else:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

            ### Fit a second order polynomial to each using `np.polyfit` ###
            if self.left_fit is not None and self.right_fit is not None:
                self.left_fit_trial = np.polyfit(lefty, leftx, 2)
                self.right_fit_trial = np.polyfit(righty, rightx, 2)
                
                left_curverad, right_curverad = self.measure_curvature_real_input(self.left_fit_trial, self.right_fit_trial, self.ploty)
                if self.check_lane_validity(left_curverad, right_curverad):
                    self.left_fit = self.left_fit_trial
                    self.right_fit = self.right_fit_trial
                    self.lane_detected = True
                else:
                    self.lane_detected = False
                    
            else: # Catch first iteration which self.left_fit is None
                self.left_fit = np.polyfit(lefty, leftx, 2)
                self.right_fit = np.polyfit(righty, rightx, 2)

            try:
                left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
                right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
            except TypeError:
                # Avoids an error if `left` and `right_fit` are still none or incorrect
                print('The function failed to fit a line!')
                left_fitx = 1*self.ploty**2 + 1*self.ploty
                right_fitx = 1*self.ploty**2 + 1*self.ploty
                self.lane_detected = False

            ## Visualization ##
            # Colors in the left and right lane regions
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            # Plots the left and right polynomials on the lane lines
            left_points = (np.asarray([left_fitx, self.ploty]).T).astype(np.int32)
            right_points = (np.asarray([right_fitx, self.ploty]).T).astype(np.int32)
            cv2.polylines(out_img, [left_points], False, (255,255,255), thickness=2)
            cv2.polylines(out_img, [right_points], False, (255,255,255), thickness=2)

        return out_img
    
    def measure_curvature_real_input(self, left_fit, right_fit, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        ##### Implement the calculation of R_curve (radius of curvature) #####
        y_eval = ym_per_pix * y_eval
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**(3/2))/np.abs(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**(3/2))/np.abs(2*right_fit[0])

        return left_curverad, right_curverad
    
    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/780 # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        ##### Implement the calculation of R_curve (radius of curvature) #####
        y_eval = ym_per_pix * y_eval
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**(3/2))/np.abs(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**(3/2))/np.abs(2*self.right_fit[0])

        return left_curverad, right_curverad
    
    def measure_center(self, img_shape):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(self.ploty)
        x_left = self.left_fit[0]*y_eval**2 + self.left_fit[1]*y_eval + self.left_fit[2]        
        x_right = self.right_fit[0]*y_eval**2 + self.right_fit[1]*y_eval + self.right_fit[2]
        center = (x_right + x_left)/2
        dist_to_center = (img_shape[1]/2 - center) * xm_per_pix
        
        return dist_to_center
    
    def check_lane_validity(self, left_curverad, right_curverad):
        return True if (abs(left_curverad - right_curverad)/(left_curverad + right_curverad) < 0.5) else False