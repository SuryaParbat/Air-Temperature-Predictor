import datetime
from functools import wraps
import os, sys, logging
from create_new_directory import create_directory

#class Logger:

def create_log_file(log_file_name):

    cwd = os.getcwd()
    folder = 'Logs'
    CHECK_FOLDER = os.path.isdir(folder)
    newPath = os.path.join(cwd, folder)

    try:

        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.mkdir(newPath)
        # folder = 'Logs'
        # filename = create_directory(log_file_name, folder)
        # '{}/{}'.format(newPath, log_file_name)
        logging.basicConfig(filename='{}/{}'.format(newPath, log_file_name), level=logging.DEBUG)
    except Exception as e:
        # logging.basicConfig(filename='{}/{}'.format(newPath, log_file_name), level=logging.DEBUG)
        return "Error while creating log file"

def moniter(function):

    try:
        @wraps(function)
        def wrapper(*args, **kwargs):
            s = datetime.datetime.now()
            print("=" * 60)
            #ip_address = "{}".format(request.remote_addr)
            #user_agent = "{}".format(request.user_agent)
            called_fuction_name = "{}".format(function.__name__)
            _ = function(*args, **kwargs)

            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            called_arguments = ",\n".join(args_repr + kwargs_repr)

            #called_arguments = dict(kwargs.items())
            called_function_arguments = '{}'.format(called_arguments)
            e = datetime.datetime.now()
            exe_time = "{}".format((e - s))
            called_function_size = "{} Bytes".format(sys.getsizeof(function))
            end_time = "{}".format(e)

            print("Start time : ", s)
            #print("IP_address : ", ip_address)
            #print("user Agent : ", user_agent)
            print("Called function Name : ", called_fuction_name)
            print("CalledFunction Arguments : ", called_function_arguments)
            print("Memory : ", called_function_size)
            print("Called function Execution Time : ", exe_time)
            print("End Time : ", end_time)
            print("=" * 60)

            message = """
                Start time : {}
                Called Fuction Name : {}
                Called Function Arguments : {}
                Memory : {}
                Execution Time : {}
                End Time : {}
                """.format(s, called_fuction_name, called_function_arguments,
                           called_function_size, exe_time, e)

            logging.debug(message)
            #logging.debug(f"function {function.__name__} called with args {signature}")

            return _

        return wrapper

    except Exception as e:
        logging.error("ERROR : {}".format(str(e)))


