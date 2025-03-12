import generic.file_system_util as fs
import hello_world.logic.hello_world_logic_module as hwl


def hello_world_execution(input_path, output_path):
    print("Executing hello_world...")
    data = fs.load_data(input_path)
    data = hwl.hello_world_prepender(data)
    fs.save_data(data, output_path)
    print("hello_world execution complete.")
