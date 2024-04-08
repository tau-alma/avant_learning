import rclpy
from mpc import AcadosMPC, SymbolicMPC, EvotorchMPC


def main(args=None):
    rclpy.init(args=args)

    node = EvotorchMPC()    

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()