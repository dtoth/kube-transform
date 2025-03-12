from execution.hello_world.logic import hello_world_logic_module as hwl


def test_hello_world_prepender():
    input_string = """A
B
C"""
    expected_output_string = """Hello World: A
Hello World: B
Hello World: C"""
    actual_output_string = hwl.hello_world_prepender(input_string)
    assert actual_output_string == expected_output_string
