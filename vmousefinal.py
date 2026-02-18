import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time
from collections import deque
import threading
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- SETUP WITH IMPROVED CONFIGURATION ---
class VirtualMouseConfig:
    # Camera settings
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    CAM_FPS = 30
    
    # Detection settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    
    # Mouse control settings
    SMOOTHENING = 5  # Lower = more responsive, Higher = smoother
    FRAME_REDUCTION = 150
    CLICK_THRESHOLD = 25
    DRAG_THRESHOLD = 35
    DOUBLE_CLICK_THRESHOLD = 40
    CLICK_COOLDOWN = 0.4
    
    # Scroll settings
    SCROLL_SPEED = 50
    SCROLL_ZONE_TOP = 0.3
    SCROLL_ZONE_BOTTOM = 0.7
    
    # Gesture stability
    GESTURE_HISTORY_SIZE = 5
    POSITION_HISTORY_SIZE = 3

class GestureStabilizer:
    """Stabilizes gesture detection by using history"""
    def __init__(self, history_size=5):
        self.history = deque(maxlen=history_size)
        self.stable_gesture = "IDLE"
        
    def update(self, new_gesture):
        self.history.append(new_gesture)
        
        # Count occurrences
        from collections import Counter
        counter = Counter(self.history)
        
        if len(self.history) == self.history.maxlen:
            # Get most common gesture
            self.stable_gesture = counter.most_common(1)[0][0]
        
        return self.stable_gesture

class PositionSmoother:
    """Smooths cursor movement"""
    def __init__(self, history_size=3):
        self.x_history = deque(maxlen=history_size)
        self.y_history = deque(maxlen=history_size)
        
    def smooth(self, x, y):
        self.x_history.append(x)
        self.y_history.append(y)
        
        if len(self.x_history) > 1:
            avg_x = sum(self.x_history) / len(self.x_history)
            avg_y = sum(self.y_history) / len(self.y_history)
            return avg_x, avg_y
        return x, y

class VirtualMouse:
    def __init__(self):
        self.config = VirtualMouseConfig()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.config.CAM_WIDTH)
        self.cap.set(4, self.config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAM_FPS)
        
        # Initialize hand detector
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
            model_complexity=1  # Better accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Screen info
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        
        # State variables
        self.drag_active = False
        self.last_click_time = 0
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        
        # Gesture stabilizers
        self.gesture_stabilizer = GestureStabilizer(self.config.GESTURE_HISTORY_SIZE)
        self.position_smoother = PositionSmoother(self.config.POSITION_HISTORY_SIZE)
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # Click state tracking
        self.left_click_ready = False
        self.right_click_ready = False
        self.double_click_ready = False
        
        print(f"✅ Virtual Mouse initialized. Screen: {self.screen_width}x{self.screen_height}")
        self.print_controls()
    
    def print_controls(self):
        """Print control instructions"""
        controls = """
🎮 VIRTUAL MOUSE CONTROLS:
========================================
1. ☝️ Only INDEX finger up → Move cursor
2. ✌️ INDEX + MIDDLE up → Left click (bring together)
3. 🤟 INDEX + MIDDLE + RING up → Right click (bring together)
4. 🤏 INDEX + THUMB touching → Drag & Drop
5. ✊ All fingers CLOSED (fist) → Scroll mode
6. 🖖 THUMB + PINKY up → Double click
========================================
Press 'q' to quit | 'r' to reset | 'd' for debug mode
"""
        print(controls)
    
    def is_finger_up(self, hand_landmarks, tip_id, pip_id, is_thumb=False):
        """Improved finger detection"""
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        
        if is_thumb:
            # For thumb, compare with index finger MCP for better accuracy
            index_mcp = hand_landmarks.landmark[5]
            return tip.x < index_mcp.x - 0.02
        else:
            # For other fingers, use more robust detection
            wrist = hand_landmarks.landmark[0]
            return tip.y < pip.y - 0.02 and tip.y < wrist.y - 0.05
    
    def get_finger_status(self, hand_landmarks):
        """Get status of all fingers"""
        finger_configs = [
            (4, 3, True),   # Thumb (tip, pip, is_thumb)
            (8, 6, False),  # Index
            (12, 10, False), # Middle
            (16, 14, False), # Ring
            (20, 18, False)  # Pinky
        ]
        
        status = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, (tip_id, pip_id, is_thumb) in enumerate(finger_configs):
            status[finger_names[i]] = self.is_finger_up(hand_landmarks, tip_id, pip_id, is_thumb)
        
        return status
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def get_landmark_coords(self, hand_landmarks, landmark_id, w, h):
        """Get coordinates of a specific landmark"""
        landmark = hand_landmarks.landmark[landmark_id]
        return int(landmark.x * w), int(landmark.y * h)
    
    def perform_scroll(self, hand_landmarks, h):
        """Improved scroll with variable speed"""
        wrist = hand_landmarks.landmark[0]
        scroll_y = wrist.y
        
        # Calculate scroll speed based on hand position (further = faster)
        if scroll_y < self.config.SCROLL_ZONE_TOP:
            speed = int(self.config.SCROLL_SPEED * 
                       (1 + (self.config.SCROLL_ZONE_TOP - scroll_y) / self.config.SCROLL_ZONE_TOP))
            pyautogui.scroll(speed)
            return f"SCROLL UP ↑ ({speed})"
        elif scroll_y > self.config.SCROLL_ZONE_BOTTOM:
            speed = int(self.config.SCROLL_SPEED * 
                       (1 + (scroll_y - self.config.SCROLL_ZONE_BOTTOM) / 
                        (1 - self.config.SCROLL_ZONE_BOTTOM)))
            pyautogui.scroll(-speed)
            return f"SCROLL DOWN ↓ ({speed})"
        
        return "SCROLL READY"
    
    def is_valid_gesture(self, finger_status):
        """Check if the current finger configuration is valid"""
        # Count number of fingers up
        fingers_up = sum(1 for v in finger_status.values() if v)
        
        # Basic validation - can't have 0 or 5 fingers up for most gestures
        # (except fist which has 0)
        if fingers_up == 5:  # All fingers up - likely detection error
            return False
        
        return True
    
    def detect_gesture(self, finger_status):
        """Detect the current gesture with validation"""
        if not self.is_valid_gesture(finger_status):
            return "INVALID"
        
        thumb, index, middle, ring, pinky = finger_status.values()
        
        # Check for fist (scroll mode)
        if not any([index, middle, ring, pinky]):  # Allow thumb to be anywhere
            return "SCROLL"
        
        # Double click (thumb + pinky)
        if thumb and pinky and not index and not middle and not ring:
            return "DOUBLE_CLICK"
        
        # Left click (index + middle)
        if index and middle and not ring and not pinky and not thumb:
            return "LEFT_CLICK"
        
        # Right click (index + middle + ring)
        if index and middle and ring and not pinky and not thumb:
            return "RIGHT_CLICK"
        
        # Move/Drag (only index)
        if index and not middle and not ring and not pinky:
            return "MOVE"
        
        return "IDLE"
    
    def handle_move_mode(self, hand_landmarks, w, h):
        """Handle mouse movement and drag"""
        cx, cy = self.get_landmark_coords(hand_landmarks, 8, w, h)
        tx, ty = self.get_landmark_coords(hand_landmarks, 4, w, h)
        
        # Check for drag gesture (thumb and index touching)
        thumb_index_dist = self.calculate_distance((cx, cy), (tx, ty))
        
        if thumb_index_dist < self.config.DRAG_THRESHOLD:
            # DRAG MODE
            if not self.drag_active:
                pyautogui.mouseDown()
                self.drag_active = True
                return "DRAG", "DRAG STARTED"
            
            # Continue dragging
            self.move_cursor(cx, cy, w, h)
            return "DRAG", "DRAGGING"
        else:
            # Normal MOVE mode
            if self.drag_active:
                pyautogui.mouseUp()
                self.drag_active = False
                return "MOVE", "DRAG ENDED"
            
            self.move_cursor(cx, cy, w, h)
            return "MOVE", ""
    
    def move_cursor(self, cx, cy, w, h):
        """Move cursor with smoothing"""
        # Convert to screen coordinates within frame reduction
        screen_x = np.interp(cx, 
                            [self.config.FRAME_REDUCTION, w - self.config.FRAME_REDUCTION], 
                            [0, self.screen_width])
        screen_y = np.interp(cy, 
                            [self.config.FRAME_REDUCTION, h - self.config.FRAME_REDUCTION], 
                            [0, self.screen_height])
        
        # Apply position smoothing
        screen_x, screen_y = self.position_smoother.smooth(screen_x, screen_y)
        
        # Apply exponential smoothing
        self.curr_x = self.prev_x + (screen_x - self.prev_x) / self.config.SMOOTHENING
        self.curr_y = self.prev_y + (screen_y - self.prev_y) / self.config.SMOOTHENING
        
        # Move mouse (with bounds checking)
        try:
            x = max(0, min(self.screen_width, int(self.curr_x)))
            y = max(0, min(self.screen_height, int(self.curr_y)))
            pyautogui.moveTo(x, y)
        except Exception as e:
            print(f"Mouse move error: {e}")
        
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
    
    def handle_left_click(self, hand_landmarks, w, h):
        """Handle left click gesture"""
        cx, cy = self.get_landmark_coords(hand_landmarks, 8, w, h)
        mx, my = self.get_landmark_coords(hand_landmarks, 12, w, h)
        
        distance = self.calculate_distance((cx, cy), (mx, my))
        
        # Visual feedback
        if distance < self.config.CLICK_THRESHOLD:
            self.left_click_ready = True
        else:
            if self.left_click_ready and time.time() - self.last_click_time > self.config.CLICK_COOLDOWN:
                pyautogui.click()
                self.last_click_time = time.time()
                self.left_click_ready = False
                return "LEFT CLICK", "CLICK!"
            self.left_click_ready = False
        
        return "LEFT CLICK READY" if self.left_click_ready else "LEFT CLICK", ""
    
    def handle_right_click(self, hand_landmarks, w, h):
        """Handle right click gesture"""
        cx, cy = self.get_landmark_coords(hand_landmarks, 8, w, h)
        mx, my = self.get_landmark_coords(hand_landmarks, 12, w, h)
        rx, ry = self.get_landmark_coords(hand_landmarks, 16, w, h)
        
        dist1 = self.calculate_distance((cx, cy), (mx, my))
        dist2 = self.calculate_distance((mx, my), (rx, ry))
        
        if dist1 < self.config.CLICK_THRESHOLD and dist2 < self.config.CLICK_THRESHOLD * 1.3:
            self.right_click_ready = True
        else:
            if self.right_click_ready and time.time() - self.last_click_time > self.config.CLICK_COOLDOWN:
                pyautogui.rightClick()
                self.last_click_time = time.time()
                self.right_click_ready = False
                return "RIGHT CLICK", "RIGHT CLICK!"
            self.right_click_ready = False
        
        return "RIGHT CLICK READY" if self.right_click_ready else "RIGHT CLICK", ""
    
    def handle_double_click(self, hand_landmarks, w, h):
        """Handle double click gesture"""
        tx, ty = self.get_landmark_coords(hand_landmarks, 4, w, h)
        px, py = self.get_landmark_coords(hand_landmarks, 20, w, h)
        
        distance = self.calculate_distance((tx, ty), (px, py))
        
        if distance < self.config.DOUBLE_CLICK_THRESHOLD:
            self.double_click_ready = True
        else:
            if self.double_click_ready and time.time() - self.last_click_time > self.config.CLICK_COOLDOWN:
                pyautogui.doubleClick()
                self.last_click_time = time.time()
                self.double_click_ready = False
                return "DOUBLE CLICK", "DOUBLE CLICK!"
            self.double_click_ready = False
        
        return "DOUBLE CLICK READY" if self.double_click_ready else "DOUBLE CLICK", ""
    
    def draw_ui(self, img, mode, action, finger_status, fps, debug=False):
        """Draw user interface"""
        h, w = img.shape[:2]
        
        # Draw control area
        cv2.rectangle(img, 
                     (self.config.FRAME_REDUCTION, self.config.FRAME_REDUCTION), 
                     (w - self.config.FRAME_REDUCTION, h - self.config.FRAME_REDUCTION), 
                     (0, 255, 0), 2)
        
        # Mode colors
        mode_colors = {
            "MOVE": (255, 0, 0),
            "DRAG": (255, 255, 0),
            "LEFT_CLICK": (0, 255, 0),
            "RIGHT_CLICK": (0, 0, 255),
            "DOUBLE_CLICK": (255, 165, 0),
            "SCROLL": (255, 0, 255),
            "IDLE": (128, 128, 128),
            "INVALID": (0, 0, 0)
        }
        
        color = mode_colors.get(mode, (255, 255, 255))
        
        # Display mode
        mode_display = mode.replace('_', ' ').title()
        cv2.putText(img, f"MODE: {mode_display}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display action
        if action:
            cv2.putText(img, action, (w//2 - 100, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display finger status
        finger_symbols = ['👍' if v else '👎' for v in finger_status.values()]
        finger_text = f"T:{finger_symbols[0]} I:{finger_symbols[1]} M:{finger_symbols[2]} R:{finger_symbols[3]} P:{finger_symbols[4]}"
        cv2.putText(img, finger_text, (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display FPS
        cv2.putText(img, f"FPS: {fps:.1f}", (w-150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Debug info
        if debug:
            y_offset = 70
            for name, value in finger_status.items():
                cv2.putText(img, f"{name}: {value}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_offset += 25
    
    def run(self):
        """Main loop"""
        debug_mode = False
        
        try:
            while True:
                # Read frame
                success, img = self.cap.read()
                if not success:
                    print("Failed to grab frame")
                    continue
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.fps_history.append(fps)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                self.last_time = current_time
                
                # Flip image for mirror effect
                img = cv2.flip(img, 1)
                h, w = img.shape[:2]
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process hands
                results = self.hands.process(img_rgb)
                
                # Default values
                mode = "IDLE"
                action = ""
                finger_status = {'thumb': False, 'index': False, 'middle': False, 
                               'ring': False, 'pinky': False}
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            img, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw_styles.get_default_hand_landmarks_style(),
                            self.mp_draw_styles.get_default_hand_connections_style()
                        )
                        
                        # Get finger status
                        finger_status = self.get_finger_status(hand_landmarks)
                        
                        # Detect gesture
                        raw_mode = self.detect_gesture(finger_status)
                        mode = self.gesture_stabilizer.update(raw_mode)
                        
                        # Handle different modes
                        if mode == "SCROLL":
                            action = self.perform_scroll(hand_landmarks, h)
                            
                            # Release drag if active
                            if self.drag_active:
                                pyautogui.mouseUp()
                                self.drag_active = False
                        
                        elif mode == "MOVE" or mode == "DRAG":
                            sub_mode, sub_action = self.handle_move_mode(hand_landmarks, w, h)
                            mode = sub_mode
                            action = sub_action
                        
                        elif mode == "LEFT_CLICK":
                            sub_mode, sub_action = self.handle_left_click(hand_landmarks, w, h)
                            mode = sub_mode
                            action = sub_action
                            
                            # Release drag if active
                            if self.drag_active:
                                pyautogui.mouseUp()
                                self.drag_active = False
                        
                        elif mode == "RIGHT_CLICK":
                            sub_mode, sub_action = self.handle_right_click(hand_landmarks, w, h)
                            mode = sub_mode
                            action = sub_action
                            
                            # Release drag if active
                            if self.drag_active:
                                pyautogui.mouseUp()
                                self.drag_active = False
                        
                        elif mode == "DOUBLE_CLICK":
                            sub_mode, sub_action = self.handle_double_click(hand_landmarks, w, h)
                            mode = sub_mode
                            action = sub_action
                            
                            # Release drag if active
                            if self.drag_active:
                                pyautogui.mouseUp()
                                self.drag_active = False
                        
                        elif mode == "INVALID":
                            # Release drag if active
                            if self.drag_active:
                                pyautogui.mouseUp()
                                self.drag_active = False
                
                else:
                    # No hand detected - release drag if active
                    if self.drag_active:
                        pyautogui.mouseUp()
                        self.drag_active = False
                
                # Draw UI
                self.draw_ui(img, mode, action, finger_status, avg_fps, debug_mode)
                
                # Show frame
                cv2.imshow("Virtual Mouse - Improved Version", img)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.drag_active = False
                    self.left_click_ready = False
                    self.right_click_ready = False
                    self.double_click_ready = False
                    print("🔄 Reset all states")
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
        
        finally:
            # Cleanup
            if self.drag_active:
                pyautogui.mouseUp()
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n👋 Virtual Mouse stopped. Goodbye!")

# Run the application
if __name__ == "__main__":
    # Safety check - warn user
    print("⚠️  Move mouse to any corner of the screen to emergency stop")
    time.sleep(2)
    
    # Create and run virtual mouse
    vm = VirtualMouse()
    vm.run()