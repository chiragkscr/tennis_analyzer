def convert_pixel_distance_meters(pixel_distance, reference_height_mtrs, reference_height_pixels):
    return (pixel_distance * reference_height_mtrs)/reference_height_pixels

def convert_meters_to_pixel_distance(meters, refernce_height_meters, reference_height_pixels):
    return (meters*reference_height_pixels)/refernce_height_meters