import rosbag2_py
import torch
from rclpy.serialization import serialize_message
from std_msgs.msg import Float64
from rosidl_runtime_py.utilities import get_message

def publish_rosbag_data(writer, topic_name, timestamp_nsec, value):
    msg = Float64()
    msg.data = float(value)
    writer.write(topic_name, serialize_message(msg), timestamp_nsec)

def make_result_rosbag(output_bag_path, datas):
    for file in datas.keys():
        storage_options = rosbag2_py.StorageOptions(uri=f"{output_bag_path}/{file}.db3", storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(output_serialization_format='cdr', input_serialization_format='cdr')
        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)

        # Create topics before writing data
        topics = set()
        for entry in datas[file].keys():
            for topic, (timestamps, values) in datas[file][entry].items():
                topic = f"{entry}_{topic}"
                if topic not in topics:
                    topics.add(topic)
                    topic_info = rosbag2_py.TopicMetadata(name=topic, type='std_msgs/msg/Float64', serialization_format='cdr')
                    writer.create_topic(topic_info)

                for timestamp, value in zip(timestamps, values):
                    timestamp_nsec = int(timestamp)
                    publish_rosbag_data(writer, topic, timestamp_nsec, value)

        writer.close()
