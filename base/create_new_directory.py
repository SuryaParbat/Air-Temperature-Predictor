import os, logging


def create_directory(file_name, folder_name):
    try:
        cwd = os.getcwd()
        CHECK_FOLDER = os.path.isdir(folder_name)
        newPath = os.path.join(cwd, folder_name)
        filename = '{}/{}'.format(newPath, file_name)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.mkdir(newPath)

        logging.debug(f"""create_directory is Successfully loaded\n""")
        logging.debug(f"""Folder : {folder_name} is Successfully created\n""")
        return filename
    except Exception as e:
        logging.error(f""" ERROR IN : create_directory : {str(e)}\n""")
        return f"""ERROR IN : create_directory : {str(e)}\n"""
