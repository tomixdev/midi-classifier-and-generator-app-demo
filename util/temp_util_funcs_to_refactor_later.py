import os


def assert_dir_path_str_format(dir_path: str):
    assert isinstance(dir_path, str), f"{dir_path = } is not a string..."

    if dir_path[-1] != "/":
        raise ValueError(f"{dir_path = }. Wrong Format...")


def assert_file_exists(file_path: str):
    assert isinstance(file_path, str), f"{file_path = } is not a string..."

    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path = }. File does not exist...")


def assert_a_dir_exists(dir_path: str):
    assert isinstance(dir_path, str), f"{dir_path = } is not a string..."

    assert_dir_path_str_format(dir_path)

    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path = }. Directory does not exist...")


def get_all_absolute_file_paths_recursively_from_directory(directory: str, file_extension_list: list[str]) -> list[str]:
    assert_dir_path_str_format(directory)

    for file_extension in file_extension_list:
        if file_extension[0] != ".":
            raise ValueError(f"{file_extension = }. Wrong Format...")

    file_paths_to_return = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(file_extension_list)):
                a_relative_path = os.path.join(root, file)
                # make a relative path into absolute path
                an_absolute_path = os.path.abspath(a_relative_path)
                file_paths_to_return.append(an_absolute_path)

    return file_paths_to_return


def get_all_class_variables_of_a_class(AClass):
    """
    :param AClass:
    :return: list

    class AClass:
        a = 1
        b = 2

    get_all_class_variables_of_a_class(AClass) # --> returns ['a', 'b']
    """

    return [attr for attr in dir(AClass) if not callable(getattr(AClass, attr)) and not attr.startswith("__")]


def duplicate_a_directory_structure_from_source_to_target(source_dir, target_dir):
    assert_dir_path_str_format(source_dir)
    assert_dir_path_str_format(target_dir)

    # convert source_dir to absolute path
    source_dir = os.path.abspath(source_dir)
    # convert target_dir to absolute path
    target_dir = os.path.abspath(target_dir)

    for root, dirs, files in os.walk(source_dir):
        for dir in dirs:
            a_relative_path = os.path.join(root, dir)
            # make a relative path into absolute path
            an_absolute_path = os.path.abspath(a_relative_path)
            # create a directory in target_dir if target dir does not exist
            if not os.path.isdir(an_absolute_path.replace(source_dir, target_dir)):
                os.makedirs(an_absolute_path.replace(
                    source_dir, target_dir), exist_ok=True)


# Simple Test Code
if __name__ == "__main__":
    class AClass:
        a = 1
        b = 2

    assert get_all_class_variables_of_a_class(AClass) == ["a", "b"]
