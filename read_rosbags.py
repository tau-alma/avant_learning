import scipy.linalg
import scipy.spatial
import rosbag2_py
import os
import numpy as np
import scipy
import avant_modeling.config as config
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ROOT_DIR = 'avant_identification_data/'
KEYS = ['x', 'y', 'theta', 'beta', 'omega', 'dot_beta', 'vx', 'steer', 'gas']

def read_bag_file(db_file):
    sensor_data = {
        'x': [],
        'y': [],
        'theta': [],
        'beta': [],
        'omega': [],
        'dot_beta': [],
        'vx': [],
        'steer': [],
        'gas': [],
    }
    timestamps = {
        'x': [],
        'y': [],
        'theta': [],
        'beta': [],
        'omega': [],
        'dot_beta': [],
        'vx': [],
        'steer': [],
        'gas': [],
        'wanted_speeds': []
    }

    storage_options = rosbag2_py.StorageOptions(uri=db_file, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    # Iterate over topics:
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        msg_type = type_map[topic]
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)

        if topic == '/wheel_odometry':
            sensor_data['x'].append(msg.pose.pose.position.x)
            timestamps['x'].append(timestamp)
            sensor_data['y'].append(msg.pose.pose.position.y)
            timestamps['y'].append(timestamp)
            sensor_data['vx'].append(msg.twist.twist.linear.x)
            timestamps['vx'].append(timestamp)            
        elif topic == '/resolver':
            sensor_data['beta'].append(msg.position[0])
            sensor_data['dot_beta'].append(msg.velocity[0])
            timestamps['beta'].append(timestamp)
            timestamps['dot_beta'].append(timestamp)
        elif topic == '/front_axle_IMU':
            sensor_data['omega'].append(msg.angular_velocity.z)
            timestamps['omega'].append(timestamp)
        elif topic == '/motion_commands':
            sensor_data['steer'].append(msg.position[1])
            sensor_data['gas'].append(msg.position[0] * msg.position[2])
            timestamps['steer'].append(timestamp)
            timestamps['gas'].append(timestamp)
        elif topic == '/wanted_speeds':
            timestamps['wanted_speeds'].append(timestamp)
        elif topic == '/GNSS_attitude/Pose':
            theta = scipy.spatial.transform.Rotation.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.y, msg.pose.pose.orientation.w])
            sensor_data['theta'].append(theta.as_euler("xyz")[2])
            timestamps['theta'].append(timestamp)

    return sensor_data, timestamps

def interpolate_sensor_data(sensor_data, timestamps, sample_rate):
    # These tell us when the autonomous operation started/stopped:
    min_timestamp = min(timestamps["wanted_speeds"])
    max_timestamp = max(timestamps["wanted_speeds"])

    # Create a common time base for interpolation
    common_time = np.arange(min_timestamp, max_timestamp, sample_rate * 1e9)

    # Interpolate the data
    interpolated_data = {}
    for key in sensor_data.keys():
        interpolated_data[key] = np.interp(common_time, timestamps[key], sensor_data[key])

    data_array = np.column_stack([interpolated_data[key] for key in KEYS])
    # Adjust common_time to start from 0
    output_time = common_time - min_timestamp

    return output_time, data_array

if __name__ == '__main__':
    for folder in os.listdir(ROOT_DIR):
        if ".csv" in folder or ".png" in folder or ".json" in folder:
            continue

        db_file = os.path.join(ROOT_DIR, folder) + f'/{folder}_0.db3'
        
        sensor_data, timestamps = read_bag_file(db_file)
        common_time, data_array = interpolate_sensor_data(sensor_data, timestamps, config.sample_rate)

        header = 'timestamp,' + ','.join(KEYS)
        np.savetxt(os.path.join(ROOT_DIR, f'{folder}.csv'), np.column_stack((common_time, data_array)), delimiter=',', header=header, comments='')
