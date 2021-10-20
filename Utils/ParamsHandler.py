'''
    Copyright:      JarvisLee
    Date:           10/21/2021
    File Name:      ParamsHandler.py
    Description:    This file is used to handle the hyperparameters.
'''

# Import the necessary library.
import argparse
from easydict import EasyDict as Config

# Set the class to encapsulate the handler's tools.
class Handler():
    '''
        This class is used to encapsulate all the functions which are used to handle the hyperparameters.\n
        This class contains four parts:\n
            - 'Convertor' is used to convert the type of each hyperparameter.
            - 'Generator' is used to generate the initial hyperparameters from the Params.txt file.
            - 'Parser' is used to parse the hyperparameters.
            - 'Displayer' is used to display the hyperparameters.
    '''
    # Set the function to convert the type of the hyperparameters.
    def Convertor(param):
        '''
            This function is used to convert the type of the hyperparameters.\n
            Params:\n
                - param: The hyperparameters.
        '''
        # Convert the hyperparameters.
        try:
            param = eval(param)
        except:
            param = param
        # Return the hyperparameters.
        return param

    # Set the function to generate the configurator of the hyperparameters.
    def Generator(paramsDir = './Params.txt'):
        '''
            This function is used to generate the configurator of hyperparameters.\n
            Params:\n
                - paramsDir: The directory of the hyperparameters' default setting file.
        '''
        # Create the configurator of hyperparameters.
        Cfg = Config()
        # Get the names of hyperparameters.
        with open(paramsDir) as file:
            lines = file.readlines()
            # Initialize the hyperparameters.
            for line in lines:
                Cfg[line.split("\n")[0].split(":")[0]] = Handler.Convertor(line.split("\n")[0].split(":")[1])
        # Return the dictionary of the hyperparameters.
        return Cfg

    # Set the function to parse the hyperparameters.
    def Parser(Cfg):
        '''
            This function is used to parse the hyperparameters.\n
            Params:\n
                - Cfg: The configurator of the hyperparameters.
        '''
        # Indicate whether the Cfg is a configurator or not.
        assert type(Cfg) is type(Config()), 'Please input the configurator.'
        # Create the hyperparameters' parser.
        parser = argparse.ArgumentParser(description = 'Hyperparameters Parser')
        # Add the hyperparameters into the parser.
        for param in Cfg.keys():
            parser.add_argument(f'-{param}', f'--{param}', f'-{param.lower()}', f'--{param.lower()}', f'-{param.upper()}', f'--{param.upper()}', dest = param, type = type(Cfg[param]), default = Cfg[param], help = f'The type of {param} is {type(Cfg[param])}')
        # Parse the hyperparameters.
        params = vars(parser.parse_args())
        # Update the configurator.
        Cfg.update(params)
        # Return the configurator.
        return Cfg
    
    # Set the function to display the hyperparameters setting.
    def Displayer(Cfg):
        '''
            This function is used to display the hyperparameters.\n
            Params:\n
                - Cfg: The configurator.
        '''
        # Indicate whether the Cfg is a configurator or not.
        assert type(Cfg) is type(Config()), 'Please input the configurator.'
        # Set the displayer.
        displayer = [''.ljust(20) + f'{param}:'.ljust(30) + f'{Cfg[param]}' for param in Cfg.keys()]
        # Return the result of the displayer.
        return "\n".join(displayer)

# Test the codes.
if __name__ == "__main__":
    print(Handler.Displayer(Handler.Parser(Handler.Generator())))