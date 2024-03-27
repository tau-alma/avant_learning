import rclpy
from mpc import AcadosMPC 


def main(args=None):
    rclpy.init(args=args)

    node = AcadosMPC()    

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()