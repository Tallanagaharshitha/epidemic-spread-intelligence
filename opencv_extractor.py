import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

class CurveExtractor:
    """Extract epidemic curve data from images using OpenCV"""
    
    def __init__(self):
        self.image = None
        self.processed_image = None
        
    def load_image(self, uploaded_file):
        """Load image from uploaded file"""
        try:
            # Read the image file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            self.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if self.image is None:
                st.error("Failed to load image. Please check the file format.")
                return None
                
            return self.image
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for curve extraction"""
        try:
            # Check if image is loaded
            if image is None:
                return None
                
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply thresholding
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            self.processed_image = cleaned
            return cleaned
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def detect_axes(self, image):
        """Detect axes in the image"""
        try:
            if image is None:
                return {'x_axis': None, 'y_axis': None}
                
            # Use Hough Lines to detect axes
            lines = cv2.HoughLinesP(
                image, 
                rho=1, 
                theta=np.pi/180, 
                threshold=50, 
                minLineLength=50, 
                maxLineGap=10
            )
            
            axes = {'x_axis': None, 'y_axis': None}
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Check if line is horizontal (potential x-axis)
                    if abs(y2 - y1) < 5 and abs(x2 - x1) > 50:
                        axes['x_axis'] = (min(y1, y2), max(y1, y2))
                    
                    # Check if line is vertical (potential y-axis)
                    if abs(x2 - x1) < 5 and abs(y2 - y1) > 50:
                        axes['y_axis'] = (min(x1, x2), max(x1, x2))
            
            return axes
        except Exception as e:
            st.error(f"Error detecting axes: {str(e)}")
            return {'x_axis': None, 'y_axis': None}
    
    def extract_curve_points(self, image, axes):
        """Extract curve points from processed image"""
        try:
            if image is None:
                return None
                
            # Find contours
            contours, _ = cv2.findContours(
                image, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Find the largest contour (likely the curve)
            if contours and len(contours) > 0:
                # Filter out very small contours
                valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
                
                if valid_contours:
                    main_curve = max(valid_contours, key=cv2.contourArea)
                    
                    # Extract points
                    points = []
                    for point in main_curve:
                        x, y = point[0]
                        points.append((x, y))
                    
                    # Sort points by x-coordinate
                    points.sort(key=lambda p: p[0])
                    
                    return points
            
            return None
        except Exception as e:
            st.error(f"Error extracting curve points: {str(e)}")
            return None
    
    def normalize_coordinates(self, points, axes, image_shape):
        """Normalize coordinates to actual data values"""
        try:
            if not points or not axes:
                return None
            
            # Get image dimensions
            height, width = image_shape[:2]
            
            # If axes not detected, use image boundaries
            x_min = axes['y_axis'][0] if axes.get('y_axis') else 0
            x_max = axes['y_axis'][1] if axes.get('y_axis') else width
            y_min = axes['x_axis'][0] if axes.get('x_axis') else 0
            y_max = axes['x_axis'][1] if axes.get('x_axis') else height
            
            # Normalize points
            normalized_data = []
            for x, y in points:
                # Normalize x (time) - range from 0 to 100
                if x_max > x_min:
                    norm_x = (x - x_min) / (x_max - x_min) * 100
                    norm_x = max(0, min(100, norm_x))
                else:
                    norm_x = 0
                
                # Normalize y (cases) - range from 0 to 100
                if y_max > y_min:
                    norm_y = (y_max - y) / (y_max - y_min) * 100
                    norm_y = max(0, min(100, norm_y))
                else:
                    norm_y = 0
                
                normalized_data.append({
                    'time_point': norm_x,
                    'cases_scaled': norm_y
                })
            
            return normalized_data
        except Exception as e:
            st.error(f"Error normalizing coordinates: {str(e)}")
            return None
    
    def create_dataframe(self, normalized_data):
        """Create a dataframe from normalized data"""
        try:
            if not normalized_data or len(normalized_data) < 2:
                return None
            
            df = pd.DataFrame(normalized_data)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['time_point'])
            
            # Generate dates
            start_date = pd.Timestamp.now() - pd.Timedelta(days=len(df))
            df['date'] = start_date + pd.to_timedelta(df['time_point'] / 100 * len(df), unit='D')
            
            # Scale cases to realistic numbers
            max_cases = 10000
            df['confirmed'] = (df['cases_scaled'] / 100 * max_cases).astype(int)
            
            # Add derived columns
            df['region'] = 'Extracted'
            df['recovered'] = (df['confirmed'] * 0.6).astype(int)
            df['deaths'] = (df['confirmed'] * 0.03).astype(int)
            df['population'] = 1000000
            
            # Calculate daily new cases
            df['new_cases'] = df['confirmed'].diff().fillna(0).astype(int)
            
            # Calculate growth rate (avoid division by zero)
            df['growth_rate'] = df['new_cases'].pct_change().fillna(0) * 100
            df['growth_rate'] = df['growth_rate'].replace([np.inf, -np.inf], 0)
            
            # Select only needed columns
            result_df = df[['date', 'region', 'confirmed', 'recovered', 'deaths', 
                           'population', 'new_cases', 'growth_rate']]
            
            return result_df
        except Exception as e:
            st.error(f"Error creating dataframe: {str(e)}")
            return None
    
    def extract_curve_data(self, image):
        """Main method to extract curve data from image"""
        try:
            if image is None:
                st.warning("No image loaded. Please upload an image first.")
                return None
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Preprocess image
            status_text.text("Step 1/4: Preprocessing image...")
            processed = self.preprocess_image(image)
            progress_bar.progress(25)
            
            if processed is None:
                status_text.text("Failed to preprocess image.")
                return None
            
            # Step 2: Detect axes
            status_text.text("Step 2/4: Detecting axes...")
            axes = self.detect_axes(processed)
            progress_bar.progress(50)
            
            # Step 3: Extract curve points
            status_text.text("Step 3/4: Extracting curve points...")
            points = self.extract_curve_points(processed, axes)
            progress_bar.progress(75)
            
            if not points or len(points) < 10:
                status_text.text("Could not detect enough curve points. Try a clearer image.")
                st.warning("Could not detect a clear curve. The image might need better contrast or clearer lines.")
                return None
            
            # Step 4: Normalize and create dataframe
            status_text.text("Step 4/4: Creating data...")
            normalized = self.normalize_coordinates(points, axes, image.shape)
            progress_bar.progress(90)
            
            if not normalized:
                status_text.text("Failed to normalize coordinates.")
                return None
            
            df = self.create_dataframe(normalized)
            progress_bar.progress(100)
            status_text.text("Extraction complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if df is not None:
                st.success(f"✅ Successfully extracted {len(df)} data points from image!")
                
                # Show preview
                with st.expander("View Extracted Data"):
                    st.dataframe(df.head(10))
            
            return df
            
        except Exception as e:
            st.error(f"Error extracting curve data: {str(e)}")
            return None
    
    def visualize_extraction(self, image):
        """Visualize the extraction process"""
        try:
            if image is None:
                return None
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')
            
            # Processed image
            processed = self.preprocess_image(image)
            if processed is not None:
                axes[1].imshow(processed, cmap='gray')
                axes[1].set_title('Processed Image', fontsize=12)
                axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, 'Processing failed', 
                           ha='center', va='center')
                axes[1].axis('off')
            
            # Extracted curve
            try:
                processed_curve = self.preprocess_image(image)
                axes_data = self.detect_axes(processed_curve)
                points = self.extract_curve_points(processed_curve, axes_data)
                
                if points:
                    result = image.copy()
                    for point in points:
                        cv2.circle(result, point, 2, (0, 255, 0), -1)
                    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    axes[2].set_title(f'Extracted Curve ({len(points)} points)', fontsize=12)
                else:
                    axes[2].text(0.5, 0.5, 'No curve detected', 
                               ha='center', va='center')
                    axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[2].axis('off')
                
            except Exception as e:
                axes[2].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                           ha='center', va='center')
                axes[2].axis('off')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error visualizing extraction: {str(e)}")
            return None