#!/usr/bin/env python3

"""
Command line utility tool for reading lidar data from an Ouster sensor and pushing to keelson as foxglove.PointCloud
"""
import time
import json
import atexit
import logging
import argparse
import warnings
from contextlib import closing

from ouster import client
import zenoh
import brefv
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud # Temporary

from environs import Env

KEELSON_INTERFACE_TYPE = "lidar"
KEELSON_TAG_POINT_CLOUD = "point_cloud"

# Reading config from environment variables
env = Env()

MQTT_BROKER_HOST: str = env("MQTT_BROKER_HOST", "example")


def run(session: zenoh.Session, args: argparse.Namespace):
    topic = brefv.construct_pub_sub_topic(
        realm=args.realm,
        entity_id=args.entity_id,
        interface_type=KEELSON_INTERFACE_TYPE,
        interface_id=args.interface_id,
        tag=KEELSON_TAG_POINT_CLOUD,
        source_id=args.source_id,
    )

    logging.info("on topic: %s", topic)

    # Declaring zenoh publisher
    publisher = session.declare_publisher(
        topic,
        priority=zenoh.Priority.INTERACTIVE_HIGH(),
        congestion_control=zenoh.CongestionControl.DROP(),
    )

    logging.info("Connecting to Ouster sensor...")

    # Connect with the Ouster sensor and start processing lidar scans
    config = client.get_config(args.ouster_hostname)
    logging.info("Sensor configuration: \n %s", config)

    logging.info("Processing packages!")

    # Connecting to Ouster sensor
    with closing(
        client.Scans.stream(args.ouster_hostname, config.udp_port_lidar, complete=True)
    ) as stream:
        # Create a look-up table to cartesian projection
        xyz_lut = client.XYZLut(stream.metadata)

        for scan in stream:
            ingress_timestamp = time.time_ns()

            # obtain destaggered xyz representation
            xyz_destaggered = client.destagger(stream.metadata, xyz_lut(scan))

            # Points as [[x, y, z], ...]
            points = xyz_destaggered.reshape(-1, xyz_destaggered.shape[-1])

            logging.debug("Type of points are: %s", points.dtype)

            data = points.tobytes()
            point_stride = len(data) / len(points)

            logging.debug("Point stride: %s", point_stride)

            # Create payload
            payload = PointCloud()
            payload.timestamp.FromNanoseconds(ingress_timestamp)

            if args.frame_id is not None:
                payload.frame_id = args.frame_id

            # Zero relative position
            payload.pose.position.x = 0
            payload.pose.position.y = 0
            payload.pose.position.z = 0

            # Identity quaternion
            payload.pose.rotation.x = 0
            payload.pose.rotation.y = 0
            payload.pose.rotation.z = 0
            payload.pose.rotation.w = 1

            # Fields are in float64 (8 bytes each)
            payload.fields.add(name="x", offset=0, type=8)
            payload.fields.add(name="y", offset=8, type=8)
            payload.fields.add(name="z", offset=16, type=8)

            payload.point_stride = int(point_stride)
            payload.data = data

            serialized_payload = payload.SerializeToString()
            logging.debug("...serialized.")

            envelope = brefv.enclose(serialized_payload)
            logging.debug("...enclosed into envelope, serialized as: %s", envelope)

            publisher.put(envelope)
            logging.info("...published to zenoh!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ouster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument(
        "--connect",
        action="append",
        type=str,
        help="Endpoints to connect to.",
    )

    parser.add_argument("-o", "--ouster-hostname", type=str, required=True)
    parser.add_argument("-r", "--realm", type=str, required=True)
    parser.add_argument("-e", "--entity-id", type=str, required=True)
    parser.add_argument("-i", "--interface-id", type=str, required=True)
    parser.add_argument("-s", "--source-id", type=str, required=True)
    parser.add_argument("-f", "--frame-id", type=str, default=None, required=True)

    ## Parse arguments and start doing our thing
    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s", level=args.log_level
    )
    logging.captureWarnings(True)
    warnings.filterwarnings("once")

    ## Construct session
    logging.info("Opening Zenoh session...")
    conf = zenoh.Config()

    if args.connect is not None:
        conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))
    session = zenoh.open(conf)

    def _on_exit():
        session.close()

    atexit.register(_on_exit)

    try:
        run(session, args)
    except KeyboardInterrupt:
        logging.info("Program ended due to user request (Ctrl-C)")
        pass