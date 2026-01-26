import os
import cv2
import numpy as np
import pandas as pd
import math
from typing import List, Tuple
import argparse
from tqdm import tqdm

# Define Point and Segment classes for geometric calculations
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Segment:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

def intersect_ray_segment(origin: Point, angle: float, segment: Segment):
    """Calculate the intersection point of a ray from origin with a line segment"""
    ray_dir_x, ray_dir_y = math.cos(angle), math.sin(angle)
    
    v1_x, v1_y = origin.x - segment.p1.x, origin.y - segment.p1.y
    v2_x, v2_y = segment.p2.x - segment.p1.x, segment.p2.y - segment.p1.y
    v3_x, v3_y = -ray_dir_y, ray_dir_x
    
    dot = v2_x * v3_x + v2_y * v3_y
    if abs(dot) < 1e-9:
        return None  # Parallel or collinear

    t1 = (v2_x * v1_y - v2_y * v1_x) / dot
    t2 = (v1_x * v3_x + v1_y * v3_y) / dot

    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return Point(origin.x + t1 * ray_dir_x, origin.y + t1 * ray_dir_y)
    
    return None

def extract_obstacles_from_image(image_path: str, threshold: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract obstacle contours (polygons) and transmittance map from input image.
    We assume areas with non-zero R channel (reflectance) are obstacles.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
        
    reflectance_map = img[:, :, 2]  # R
    transmittance_map = img[:, :, 1]  # G
    
    # Binarization: treat anything above threshold as obstacle
    _, binary_mask = cv2.threshold(reflectance_map, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    # cv2.RETR_EXTERNAL only detects the outermost contours, which is exactly what we want
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, transmittance_map

def generate_visibility_boundary(
    obstacles_contours: List[np.ndarray],
    tx_position: Tuple[float, float],
    img_shape: Tuple[int, int],
    transmittance_map: np.ndarray
) -> np.ndarray:
    """
    Generate geometric boundary lines of visible areas using a new method based on transmittance map
    for point extraction and ray generation. White lines (255) represent visibility boundaries,
    black background (0) represents other areas.
    """
    height, width = img_shape
    tx = Point(tx_position[1], tx_position[0])

    
    special_points = []
    directions = {}  

    for y in range(height):
        for x in range(width):


            # Get neighboring pixel values
            left = transmittance_map[y, x-1] if x > 0 else 0
            right = transmittance_map[y, x+1] if x < width-1 else 0
            up = transmittance_map[y-1, x] if y > 0 else 0
            down = transmittance_map[y+1, x] if y < height-1 else 0

            left_up = transmittance_map[y-1, x-1] if x > 0 and y > 0 else 0
            right_up = transmittance_map[y-1, x+1] if x < width-1 and y > 0 else 0
            left_down = transmittance_map[y+1, x-1] if x > 0 and y < height-1 else 0
            right_down = transmittance_map[y+1, x+1] if x < width-1 and y < height-1 else 0

            left_left = transmittance_map[y, x-2] if x > 1 else 0
            right_right = transmittance_map[y, x+2] if x < width-2 else 0
            up_up = transmittance_map[y-2, x] if y > 1 else 0
            down_down = transmittance_map[y+2, x] if y < height-2 else 0

            
            
            # Determine relative position of tx
            is_ne_sw = (tx.x >= x and tx.y <= y) or (tx.x < x and tx.y > y)  # Northeast or Southwest
            is_nw_se = (tx.x <= x and tx.y < y) or (tx.x > x and tx.y >= y)  # Northwest or Southeast

            if transmittance_map[y, x] == 0:
                # if is_ne_sw:
                #     if (left == 0 and up == 0 and down != 0 and down_down !=0 and right != 0 and right_right !=0) or (left != 0 and left_left !=0 and up != 0 and up_up !=0 and down == 0 and right == 0):
                #         special_points.append(Point(x, y))
                #         continue
                #     else:
                #         continue
                # elif is_nw_se:
                #     if (left == 0 and down == 0 and up != 0 and up_up !=0 and right != 0 and right_right !=0) or (left != 0 and left_left !=0 and down != 0 and down_down !=0 and up == 0 and right == 0):
                #         special_points.append(Point(x, y))
                #         continue
                #     else:
                #         continue
                # else:
                continue
 # However, we later found that some rectangle vertices are themselves 0
            
            # First type: rectangle vertices
            if is_ne_sw:
                if (left == 0 and up == 0 and down != 0 and down_down !=0 and right != 0 and right_right !=0) or (left != 0 and left_left !=0 and up != 0 and up_up !=0 and down == 0 and right == 0):
                    special_points.append(Point(x, y))
                    continue
            elif is_nw_se:
                if (left == 0 and down == 0 and up != 0 and up_up !=0 and right != 0 and right_right !=0) or (left != 0 and left_left !=0 and down != 0 and down_down !=0 and up == 0 and right == 0):
                    special_points.append(Point(x, y))
                    continue

            # Second type: rectangle vertices with high transmittance (direction-independent)
            if transmittance_map[y, x] >= 6:
                if (left != 0 and left_left !=0 and up != 0 and up_up !=0 and down == 0 and right == 0) or \
                (left != 0 and left_left !=0 and down != 0 and down_down !=0 and up == 0 and right != 0) or \
                (right != 0 and right_right !=0 and up != 0 and up_up !=0 and down == 0 and left == 0) or \
                (right != 0 and right_right !=0 and down != 0 and down_down !=0 and up == 0 and left == 0):
                    special_points.append(Point(x, y))
                    continue

            # Third type
            # Up and down are 0, one of left/right is greater than current, other is equal
            if up == 0 and down == 0:
                if (left > transmittance_map[y, x] and right == transmittance_map[y, x] and left_up ==0 and tx.y <=y) or \
                (left > transmittance_map[y, x] and right == transmittance_map[y, x] and left_down ==0 and tx.y >=y) or \
                (right > transmittance_map[y, x] and left == transmittance_map[y, x] and right_up ==0 and tx.y <=y) or \
                (right > transmittance_map[y, x] and left == transmittance_map[y, x] and right_down ==0 and tx.y >=y):
                    special_points.append(Point(x, y))
                    continue
            # Left and right are 0, one of up/down is greater than current, other is equal
            if left == 0 and right == 0:
                if (up > transmittance_map[y, x] and down == transmittance_map[y, x] and left_up ==0 and tx.x <=x) or \
                (up > transmittance_map[y, x] and down == transmittance_map[y, x] and right_up ==0 and tx.x >=x) or \
                (down > transmittance_map[y, x] and up == transmittance_map[y, x] and left_down ==0 and tx.x <=x) or \
                (down > transmittance_map[y, x] and up == transmittance_map[y, x] and right_down ==0 and tx.x >=x):
                    special_points.append(Point(x, y))
                    continue

            # Fourth type: two consecutive rows of third type points
            if (up == 0 and down == transmittance_map[y, x] and tx.y <=y) or (up == transmittance_map[y, x] and down == 0 and tx.y >=y):
                if (left > transmittance_map[y, x] and right == transmittance_map[y, x] and left_up ==0 and tx.y <=y) or \
                (left > transmittance_map[y, x] and right == transmittance_map[y, x] and left_down ==0 and tx.y >=y) or \
                (right > transmittance_map[y, x] and left == transmittance_map[y, x] and right_up ==0 and tx.y <=y) or \
                (right > transmittance_map[y, x] and left == transmittance_map[y, x] and right_down ==0 and tx.y >=y):
                    special_points.append(Point(x, y))
                    continue
            
            if (left == 0 and right == transmittance_map[y, x] and tx.x <=x) or (left == transmittance_map[y, x] and right == 0 and tx.x >=x):
                if (up > transmittance_map[y, x] and down == transmittance_map[y, x] and left_up ==0 and tx.x <=x) or \
                (up > transmittance_map[y, x] and down == transmittance_map[y, x] and left_down ==0 and tx.x >=x) or \
                (down > transmittance_map[y, x] and up == transmittance_map[y, x] and right_up ==0 and tx.x <=x) or \
                (down > transmittance_map[y, x] and up == transmittance_map[y, x] and right_down ==0 and tx.x >=x):
                    special_points.append(Point(x, y))
                    continue

    # Create boundary map
    boundary_map = np.zeros((height, width), dtype=np.uint8)

    # For each special point, draw a line along the ray direction (excluding the segment from tx to point)
    for pt in special_points:
        # Calculate direction from tx to pt
        dx = pt.x - tx.x
        dy = pt.y - tx.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0:
            continue
        dir_x = dx / dist
        dir_y = dy / dist

        # Calculate the far end point of the ray to the image boundary
        t = 1e6  # Large number to ensure extension beyond the image
        end_x = pt.x + t * dir_x
        end_y = pt.y + t * dir_y

        # Draw from pt to end, OpenCV will automatically clip to image boundary
        cv2.line(boundary_map, (int(pt.x), int(pt.y)), (int(end_x), int(end_y)), 255, 1)

    # Add circles: draw circles with specified radii centered at tx
    # radii = [9, 12, 17, 19, 21, 24, 27, 30, 34, 38, 43, 48, 54, 61, 68, 76, 86, 97, 108, 122, 137, 151]
    # center = (int(tx.x), int(tx.y))  # Note: in OpenCV, (x, y) is (col, row)
    # for r in radii:
    #     cv2.circle(boundary_map, center, r, 255, 1)  

    return boundary_map

def main(args):
    """Main execution function"""
    # Check if paths exist
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist -> {args.input_dir}")
        return
    if not os.path.isdir(args.positions_dir):
        print(f"Error: Positions directory does not exist -> {args.positions_dir}")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all input png files
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png')]
    
    if not input_files:
        print(f"Warning: No .png files found in input directory {args.input_dir}")
        return

    print(f"Found {len(input_files)} input files. Starting processing...")

    # Use tqdm to create progress bar
    for filename in tqdm(input_files, desc="Generating boundary maps"):
        base_name = os.path.splitext(os.path.basename(filename))[0]  
        
        # Extract prefix and sample_id
        parts = base_name.split('_')
        if len(parts) != 4 or not parts[3].startswith('S'):
            tqdm.write(f"Warning: Invalid filename format {filename}, skipped.")
            continue
        prefix = '_'.join(parts[:3])  # 'B1_Ant1_f1'
        sample_str = parts[3][1:]  # '0' from S0
        try:
            sample_id = int(sample_str)
        except ValueError:
            tqdm.write(f"Warning: Unable to parse sample ID from {filename}, skipped.")
            continue
        
        # Build corresponding CSV path {prefix}.csv
        csv_path = os.path.join(args.positions_dir, f"Positions_{prefix}.csv")
        
        if not os.path.isfile(csv_path):
            # tqdm.write is thread-safe printing method
            tqdm.write(f"Warning: CSV file {csv_path} not found for {base_name}, skipped.")
            continue

        # 读取CSV
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        tx_df = None
        for encoding in encodings_to_try:
            try:
                tx_df = pd.read_csv(csv_path, encoding=encoding)
                tqdm.write(f"Successfully read CSV file {csv_path} using encoding {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if tx_df is None:
            tqdm.write(f"Error: Unable to read CSV file {csv_path}")
            continue

        # Assume columns 'X', 'Y', 'Azimuth', ignore 'Azimuth'
        if 'X' not in tx_df.columns or 'Y' not in tx_df.columns:
            tqdm.write(f"Error: CSV file {csv_path} missing 'X' or 'Y' columns")
            continue

        # Check if sample_id is within range
        if sample_id >= len(tx_df):
            tqdm.write(f"Warning: sample_id {sample_id} exceeds number of rows {len(tx_df)} in CSV {csv_path}, skipped.")
            continue
        
        # Get corresponding row
        row = tx_df.iloc[sample_id]
        tx_pos = (row['X'], row['Y'])
        
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, f"boundary_{base_name}.png")  # 保持包括 _S{id}

        try:
            # Extract obstacle contours and transmittance map
            obstacles, transmittance_map = extract_obstacles_from_image(input_path)
            
            # Get image dimensions and transmitter position
            img = cv2.imread(input_path)
            img_h, img_w = img.shape[:2]

            # Generate visibility boundary
            boundary_map = generate_visibility_boundary(obstacles, tx_pos, (img_h, img_w), transmittance_map)
            
            # Save results
            cv2.imwrite(output_path, boundary_map)

        except Exception as e:
            tqdm.write(f"Error processing file {filename}: {e}")

    print("\nProcessing complete!")
    print(f"Generated boundary maps have been saved to: {args.output_dir}")


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate geometric boundary lines of visible areas based on input images and transmitter positions.")
    parser.add_argument('--input-dir', type=str, required=True, help="Path to folder containing input PNG images.")
    parser.add_argument('--positions-dir', type=str, required=True, help="Path to folder containing transmitter position CSV files, one CSV per input image.")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to folder for saving generated boundary maps.")
    
    # Update default parameters to match new settings
    # args = parser.parse_args([
    #     '--input-dir', './ICASSP2025_Dataset/Inputs/Task_1_ICASSP',
    #     '--positions-dir', './ICASSP2025_Dataset/Positions',
    #     '--output-dir', './BoundaryMaps'
    # ])
    args = parser.parse_args([
        '--input-dir', 'ICASSP2025_Dataset/ICASSP2025_Dataset/Inputs/Task_1_ICASSP',
        '--positions-dir', 'ICASSP2025_Dataset/ICASSP2025_Dataset/Positions',
        '--output-dir', './BoundaryMaps'
    ])

    args = parser.parse_args()
    main(args)