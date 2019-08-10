import json
import math
import sys
sys.path.append("/opt/")

from sympy import *
from sympy.geometry import *
from sympy import symbols
from sympy.plotting import plot

from PIL import Image, ImageOps, ImageDraw
import numpy as np
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

full_range = 500
color_print_threshold = 100
fail_reward = 2e-26
minimum_reward = 1
max_steering_angle = 30
steering_granularity = 5

max_speed = 8
speed_granularity = 1


def lambda_handler(event, context):
    params = json.loads(event["queryStringParameters"]["json"])
    print(params)
    filename = "mappath.png"
    with Image.open(filename) as full_track:
        draw = ImageDraw.Draw(full_track)
        p = convert_simulation_to_imagePoint(params['x'], params['y'])
        r = 5
        draw.ellipse((p.x - r, p.y - r, p.x + r, p.y + r), fill=(0, 0, 0, 255))
        full_track.save("/tmp/car.png")

        circle = get_car_heading_view_circle(params, full_track)
        circle.save('/tmp/circle.png')

        color_points = get_color_points(circle)
        number_of_color_point = sum(map(len, color_points))
        print(number_of_color_point)

        # No rewards if car does not cover the minimum side of line!
        if (number_of_color_point < color_print_threshold
                or params['is_reversed'] or not params['all_wheels_on_track']):
            print("Fail case!")
            return {
                'statusCode': 200,
                'body': json.dumps({"reward": float(fail_reward)})
            }

        print_iterator(map(len, color_points))

        regression_ray = get_linear_regression_ray(color_points, params)
        center = Point(full_range / 2, full_range / 2)
        perpendicular_distance = float(regression_ray.distance(center).evalf())
        print("perpendicular distance: " + str(perpendicular_distance))

        target_direction = math.degrees(atan(regression_ray.slope))
        steering_angle = params['steering_angle']
        steering_ray = Ray(Point(full_range / 2, full_range / 2),
                           angle=math.radians(steering_angle))
        angle_diff = target_direction - params['steering_angle']

        # Read input parameters
        track_width = params['track_width']

        distance_reward = gaussian(perpendicular_distance, 0,
                                   full_range / 4) * 10
        print("distance reward: " + str(distance_reward))
        steering_angle = params['steering_angle']
        color_ratio = get_color_ratio(color_points)

        print("angle diff: ", angle_diff)
        print("steering angle: ", steering_angle)
        print("color_ratio", color_ratio)

        track_reward =  get_red_reward(angle_diff, steering_angle , color_ratio[0]) \
                        + get_green_reward(angle_diff, steering_angle, color_ratio[1]) \
                        + get_blue_reward(angle_diff, steering_angle, color_ratio[2])
        track_reward = track_reward
        print("track reward:" + str(track_reward))

        speed_reward = params['speed']
        speed_reward = (speed_reward / max_speed * 100)
        print("speed reward:" + str(speed_reward))

        reward = distance_reward + track_reward + speed_reward

        progress = params['progress']
        if progress > 70 and progress < 100:
            reward = reward * (1 + progress / 100)
        elif progress >= 100:
            reward = reward * 5

        print("reward: " + str(reward))

    return {'statusCode': 200, 'body': json.dumps({"reward": float(reward)})}


def print_iterator(it):
    for x in it:
        print(x, end=' ')
    print('')  # for new line


def convert_simulation_to_imagePoint(x, y):
    return Point2D((2467 - 606) * x / 8, (1313 - 187) * (5 - y) / 5,
                   evaluate=False)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def get_car_heading_view_circle(params, full_track):
    p = convert_simulation_to_imagePoint(params['x'], params['y'])
    size = (full_range, full_range)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    area = (p.x - full_range / 2, p.y - full_range / 2, p.x + full_range / 2,
            p.y + full_range / 2)
    local = full_track.crop(area)
    circle = ImageOps.fit(local, mask.size, centering=(0.5, 0.5))
    circle.putalpha(mask)
    return circle.rotate(-params['heading'])


def get_color_points(circle):
    red = (255, 0, 0, 255)
    green = (0, 255, 0, 255)
    blue = (0, 0, 255, 255)
    red_points = []
    green_points = []
    blue_points = []
    color_set = set()

    for iy in range(circle.height):
        for x in range(int(circle.width / 2), circle.width):
            y = circle.height - iy - 1
            pixel = circle.getpixel((x, iy))
            color_set.add(pixel)
            if pixel == red:
                red_points.append((x, y))
            elif pixel == green:
                green_points.append((x, y))
            elif pixel == blue:
                blue_points.append((x, y))

    return [red_points, green_points, blue_points]


def get_color_ratio(color_points):
    number_of_color_point = sum(map(len, color_points))
    return list(map(lambda x: len(x) / number_of_color_point, color_points))


def get_red_reward(angle_diff, steering_angle, ratio):
    # No turn right, and should be positive!
    if steering_angle > (max_steering_angle / steering_granularity):
        return fail_reward
    elif abs(angle_diff) < (max_steering_angle -
                            max_steering_angle / steering_granularity):
        red_award = (max_steering_angle / abs(angle_diff)) * ratio
        print("red award: ", red_award)
        return red_award
    else:
        return fail_reward


def get_green_reward(angle_diff, steering_angle, ratio):
    if abs(angle_diff) > (max_steering_angle -
                          max_steering_angle / steering_granularity):
        return fail_reward
    else:
        green_award = (max_steering_angle / abs(angle_diff)) * ratio
        print("green award: ", green_award)
        return green_award


def get_blue_reward(angle_diff, steering_angle, ratio):
    # No turn right, and should be positive!
    if steering_angle > (max_steering_angle / steering_granularity):
        return fail_reward
    elif angle_diff < (max_steering_angle -
                       max_steering_angle / steering_granularity):
        blue_award = (max_steering_angle / abs(angle_diff)) * ratio
        print("blue award: ", blue_award)
        return blue_award
    else:
        return fail_reward


def get_x_y_sequence(points):
    x = np.array(list(map(lambda t: t[0], points))).reshape((-1, 1))
    y = np.array(list(map(lambda t: t[1], points)))
    return x, y


def get_linear_regression_ray(color_points, params):
    points = sum(color_points, [])
    x, y = get_x_y_sequence(points)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_[0])
    print('angle in degree:', math.degrees(math.atan(model.coef_[0])))
    return Ray(Point(0, model.intercept_), angle=atan(model.coef_[0]))
