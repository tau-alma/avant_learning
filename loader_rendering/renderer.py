import os
import pygame
import numpy as np
import cv2


def rotate_image(image, angle, p_rot):
    w, h = image.get_size()
    large_surface = pygame.Surface((2*w, 2*h), pygame.SRCALPHA)
    blit_pos = (large_surface.get_width() // 2 - p_rot[0], large_surface.get_height() // 2 - p_rot[1])
    large_surface.blit(image, blit_pos)
    rotated_large_surface = pygame.transform.rotate(large_surface, -angle)
    center_of_rotated = rotated_large_surface.get_rect().center
    return rotated_large_surface, center_of_rotated

def grayscale(surface):
    width, height = surface.get_size()
    buffer = pygame.image.tostring(surface, 'RGBA')
    arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    gray = rgb.mean(axis=2).astype(np.uint8)
    gray_rgb = np.stack([gray]*3, axis=-1)
    gray_rgba = np.dstack([gray_rgb, alpha])
    return pygame.image.frombuffer(gray_rgba.tobytes(), (width, height), 'RGBA')

class LoaderRenderer:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BROWN = (150, 75, 0)

    HALF_MACHINE_WIDTH = 0.6
    HALF_MACHINE_LENGTH = 1.3
    MACHINE_RADIUS = 0.8

    def __init__(self, render_resolution, view_distance, store_frames=False):
        self.render_resolution = render_resolution
        self.view_distance = view_distance
        pos_to_pixel_scaler = self.render_resolution / (4*view_distance)
        self.store_frames = store_frames
        self.frames = []

        pygame.init()
        self.screen = pygame.Surface((self.render_resolution, self.render_resolution))
        package_dir = os.path.dirname(__file__)
        # Construct paths relative to this file
        front_image = pygame.image.load(os.path.join(package_dir, 'assets', 'front.png'))
        rear_image = pygame.image.load(os.path.join(package_dir, 'assets', 'rear.png'))
        front_gray_image = grayscale(front_image)
        rear_gray_image = grayscale(rear_image)
        avant_image_pixel_scaler = np.mean([452 / self.HALF_MACHINE_LENGTH, 428 / self.HALF_MACHINE_LENGTH])
        avant_scale_factor = pos_to_pixel_scaler / avant_image_pixel_scaler
        self.front_center_offset = np.array([215, 430]) * avant_scale_factor
        self.rear_center_offset = np.array([226, 0]) * avant_scale_factor
        self.front_image = pygame.transform.scale(front_image, (avant_scale_factor*front_image.get_width(), avant_scale_factor*front_image.get_height()))
        self.rear_image = pygame.transform.scale(rear_image, (avant_scale_factor*rear_image.get_width(), avant_scale_factor*rear_image.get_height()))
        self.front_gray_image = pygame.transform.scale(front_gray_image, (avant_scale_factor*front_gray_image.get_width(), avant_scale_factor*front_gray_image.get_height()))
        self.rear_gray_image = pygame.transform.scale(rear_gray_image, (avant_scale_factor*rear_gray_image.get_width(), avant_scale_factor*rear_gray_image.get_height()))
    
    def render_video(self, filename, fps=30) -> np.ndarray:
        """ Saves the collected frames as .mp4

        Args:
            filename (str): name of the file (e.g. "vid.mp4")
            fps (int): video framerate
            
        Returns:
            np.ndarray: NxRxRx3 array of frames
        """
        frames = np.asarray(self.frames)
        height, width, layers = frames[0].shape
        video_filename = filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(bgr_frame)
        video.release()
        self.frames = []
        return frames

    def render_frame(self, state: np.ndarray, goal: np.ndarray, horizon: np.ndarray = [], obstacles: np.ndarray = [], comparision: np.ndarray = []) -> np.ndarray:
        """ Draws the scenario as a numpy array

        Args:
            state (np.ndarray): 4x1 numpy array of form [x_pos, y_pos, heading, center_angle]
            goal (np.ndarray): 3x1 numpy array of form [x_pos, y_pos, heading]
            horizon (np.ndarray, optional): Nx4 numpy array, where N is the number of future predictions of form [x_pos, y_pos, heading, center_angle]
            obstacles (np.ndarray, optional): Mx3 numpy array, where M is the number of (circular) obstacles of form [x_pos, y_pos, radius]
            comparision (np.ndarray, optional): Kx4 numpy array, where K is a target trajectory consisting of [x_pos, y_pos, radius, center_angle]

        Returns:
            np.ndarray: The resulting frame 
        """
        x_f = state[0]
        y_f = state[1]
        theta_f = state[2]
        beta = state[3]
        x_goal = goal[0]
        y_goal = goal[1]
        theta_goal = goal[2]  

        center = self.render_resolution // 2
        pos_to_pixel_scaler = self.render_resolution / (4*self.view_distance)

        surf = pygame.Surface((self.render_resolution, self.render_resolution))
        surf.fill(self.WHITE)

        alpha_surf = pygame.Surface((self.render_resolution, self.render_resolution))
        alpha_surf.set_alpha(72)
        alpha_surf.fill(self.WHITE)

        grid_spacing_px = int(pos_to_pixel_scaler * 1.0)
        grid_color = (0, 0, 0, 30)
        grid_surface = pygame.Surface((self.render_resolution, self.render_resolution), pygame.SRCALPHA)
        for x in range(0, self.render_resolution, grid_spacing_px):
            pygame.draw.line(grid_surface, grid_color, (x, 0), (x, self.render_resolution))
        for y in range(0, self.render_resolution, grid_spacing_px):
            pygame.draw.line(grid_surface, grid_color, (0, y), (self.render_resolution, y))
        surf.blit(grid_surface, (0, 0))

        # Visualizing avant front collision bound:
        pygame.draw.circle(alpha_surf, self.RED, 
                           center=(center + pos_to_pixel_scaler*(x_f), center + pos_to_pixel_scaler*(y_f)),
                           radius=pos_to_pixel_scaler*self.MACHINE_RADIUS)
        
        # Visualizing avant rear collision bound:
        x_off = pos_to_pixel_scaler*(
            x_f 
            - np.cos(theta_f) * np.cos(beta/4)*self.HALF_MACHINE_LENGTH/2 
            - np.sin(theta_f) * np.sin(beta/4)*self.HALF_MACHINE_LENGTH/2 
            - np.cos(-theta_f - beta) * self.HALF_MACHINE_LENGTH/2 
        )
        y_off = pos_to_pixel_scaler*(
            y_f 
            - np.sin(theta_f) * np.cos(beta/4)*self.HALF_MACHINE_LENGTH/2 
            + np.cos(theta_f) * np.sin(beta/4)*self.HALF_MACHINE_LENGTH/2 
            + np.sin(-theta_f - beta) * self.HALF_MACHINE_LENGTH/2 
        )
        pygame.draw.circle(alpha_surf, self.RED, 
                           center=(center + x_off, center + y_off),
                           radius=pos_to_pixel_scaler*self.MACHINE_RADIUS)

        # Black magic to shift the avant frame images correctly given the kinematics:
        x_off = pos_to_pixel_scaler*(
            x_goal - np.cos(theta_goal) * self.HALF_MACHINE_LENGTH/2
        )
        y_off = pos_to_pixel_scaler*(
            y_goal - np.sin(theta_goal) * self.HALF_MACHINE_LENGTH/2 
        )
        rotated_image, final_position = rotate_image(self.rear_gray_image, np.rad2deg(theta_goal + np.pi/2), self.rear_center_offset.tolist())
        surf.blit(rotated_image, 
                       (center + x_off - final_position[0], 
                        center + y_off - final_position[1]))
        rotated_image, final_position = rotate_image(self.front_gray_image, np.rad2deg(theta_goal + np.pi/2), self.front_center_offset.tolist())
        surf.blit(rotated_image, 
                       (center + x_off - final_position[0], 
                        center + y_off - final_position[1]))
        pygame.draw.circle(surf, self.RED, 
                           center=(center + pos_to_pixel_scaler*(x_goal), center + pos_to_pixel_scaler*(y_goal)),
                           radius=5)

        # During evaluation, draw the obstacles, if provided:
        for j in range(len(obstacles)):
            x_o, y_o, r_o = obstacles[j]
            pygame.draw.circle(surf, self.BLACK,
                            center=(center + pos_to_pixel_scaler*(x_o), center + pos_to_pixel_scaler*(y_o)),
                            radius=pos_to_pixel_scaler*r_o)
            
        # During evaluation, draw the comparision, if provided:
        for j in range(len(comparision)):
            x_f_val, y_f_val, theta_f_val, beta_val = comparision[j, :4]
            draw_surf = alpha_surf
            # Black magic to shift the image correctly given the kinematics:
            x_off = pos_to_pixel_scaler*(
                x_f_val - np.cos(theta_f_val) * np.cos(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
                - np.sin(theta_f_val) * np.sin(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
            )
            y_off = pos_to_pixel_scaler*(
                y_f_val - np.sin(theta_f_val) * np.cos(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
                + np.cos(theta_f_val) * np.sin(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
            )
            rotated_image, final_position = rotate_image(self.rear_gray_image, np.rad2deg(theta_f_val + beta_val + np.pi/2), self.rear_center_offset.tolist())
            draw_surf.blit(rotated_image, 
                        (center + x_off - final_position[0], 
                         center + y_off - final_position[1]))
            rotated_image, final_position = rotate_image(self.front_gray_image, np.rad2deg(theta_f_val + np.pi/2), self.front_center_offset.tolist())
            draw_surf.blit(rotated_image, 
                        (center + x_off - final_position[0], 
                         center + y_off - final_position[1]))
            pygame.draw.circle(draw_surf, self.RED,
                            center=(center + pos_to_pixel_scaler*(x_f_val), center + pos_to_pixel_scaler*(y_f_val)),
                            radius=5)

        # During evaluation, draw the prediction horizon, if provided:
        if len(horizon):
            data = np.r_[np.c_[x_f, y_f, theta_f, beta],
                         horizon]
        else:
            data = np.c_[x_f, y_f, theta_f, beta]
        for j in range(len(data)):
            x_f_val, y_f_val, theta_f_val, beta_val = data[j]
            if j == 0:
                draw_surf = surf
            else:
                draw_surf = alpha_surf
            # Black magic to shift the image correctly given the kinematics:
            x_off = pos_to_pixel_scaler*(
                x_f_val - np.cos(theta_f_val) * np.cos(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
                - np.sin(theta_f_val) * np.sin(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
            )
            y_off = pos_to_pixel_scaler*(
                y_f_val - np.sin(theta_f_val) * np.cos(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
                + np.cos(theta_f_val) * np.sin(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
            )
            rotated_image, final_position = rotate_image(self.rear_image, np.rad2deg(theta_f_val + beta_val + np.pi/2), self.rear_center_offset.tolist())
            draw_surf.blit(rotated_image, 
                        (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))
            rotated_image, final_position = rotate_image(self.front_image, np.rad2deg(theta_f_val + np.pi/2), self.front_center_offset.tolist())
            draw_surf.blit(rotated_image, 
                        (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))
            pygame.draw.circle(draw_surf, self.RED,
                               center=(center + pos_to_pixel_scaler*(x_f_val), center + pos_to_pixel_scaler*(y_f_val)),
                               radius=5)
        
        self.screen.blits([
            (surf, (0, 0)),
            (alpha_surf, (0, 0))
        ])
        buffer = pygame.transform.flip(self.screen, True, False)
        buffer = pygame.surfarray.array3d(buffer)
        if self.store_frames:
            self.frames.append(buffer)

        return buffer