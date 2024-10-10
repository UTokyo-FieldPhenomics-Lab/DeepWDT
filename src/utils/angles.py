import math

def correct_angle(angle, df_id):
    if angle<0:
        angle+=math.pi

    if 0 <= angle < (1/4)*math.pi:
        x_diff = df_id.iloc[-1]['center_x'] - df_id.iloc[0]['center_x']
        angle += math.pi/2 if x_diff > 0 else -math.pi/2

    elif (1/4)*math.pi <= angle < (1/2)*math.pi:
        y_diff = df_id.iloc[-1]['center_y'] - df_id.iloc[0]['center_y']
        angle += math.pi/2 if y_diff > 0 else -math.pi/2
            
    elif math.pi/2 <= angle < (3/4)*math.pi:
        y_diff = df_id.iloc[-1]['center_y'] - df_id.iloc[0]['center_y']
        angle += -math.pi/2-math.pi if y_diff > 0 else -math.pi/2

    else:
        x_diff = df_id.iloc[-1]['center_x'] - df_id.iloc[0]['center_x']
        angle += -3*math.pi/2 if x_diff <= 0 else -math.pi/2

    return angle