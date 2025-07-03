import os,sys,warnings
import argparse
import pkg_resources
import subprocess

warnings.filterwarnings('ignore')

class Config:
    """Singleton configuration class for FEX-DM"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_paths()
            self._setup_packages()
            self._setup_arguments()
            Config._initialized = True

    def _setup_paths(self):
        """Setup all path configurations"""
        # Project directory structure
        self.DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Example directory structure
        self.DIR_TRIAD = os.path.join(self.DIR_PROJECT, 'Example', 'MC_triad')
        self.DIR_EQUIPART = os.path.join(self.DIR_PROJECT, 'Results', 'equipart')
        self.DIR_CASCADE = os.path.join(self.DIR_PROJECT, 'Results', 'cascade')

        # Store file paths instead of loading numpy arrays immediately
        self.TRIAD_MODEL_CONFIG = {
            '1':{
                'name': 'FEX_dim_1',
                'op_seq_equipart_path': os.path.join(self.DIR_EQUIPART, 'optimal_idx_1.npy'),
                'op_seq_cascade_path': os.path.join(self.DIR_CASCADE, 'optimal_idx_1.npy')
            },
            '2':{
                'name': 'FEX_dim_2',
                'op_seq_equipart_path': os.path.join(self.DIR_EQUIPART, 'optimal_idx_2.npy'),
                'op_seq_cascade_path': os.path.join(self.DIR_CASCADE, 'optimal_idx_2.npy')
            },
            '3':{
                'name': 'FEX_dim_3',
                'op_seq_equipart_path': os.path.join(self.DIR_EQUIPART, 'optimal_idx_3.npy'),
                'op_seq_cascade_path': os.path.join(self.DIR_CASCADE, 'optimal_idx_3.npy')
            }
        }
    
    def _setup_packages(self):
        """Setup package management configurations"""
        # Required packages and their versions
        self.REQUIRED_PACKAGES = {
            'faiss-cpu': '1.8.1',
            'sympy': '1.13.1',
            'torch': '2.3.1',
            'torchvision': '0.18.1',
            'numpy': '1.26.4',
            'matplotlib': '3.8.0',
            'scipy': '1.13.0',
            'numba': '0.59.0',
            'scikit-learn': '1.3.2',
        }
        
        # Built-in modules that don't need installation
        self.BUILTIN_MODULES = {
            'typing',  # Built into Python 3.5+
            'dataclasses',  # Built into Python 3.7+
        }
    
    def _setup_arguments(self):
        """Setup argument parser"""
        pass

    def check_and_install_packages(self):
        """Check and install required packages"""
        print('Checking and installing required packages...')
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        missing_packages = []
        outdated_packages = []

        # Check built-in modules first
        for module in self.BUILTIN_MODULES:
            try:
                __import__(module)
                print(f"[SUCCESS] {module} (built-in) is available")
            except ImportError:
                print(f"[WARNING] {module} (built-in) is not available in this Python version")

        # Check external packages
        for package, version in self.REQUIRED_PACKAGES.items():
            if package not in installed_packages:
                missing_packages.append(package)
            else:
                # check if version is sufficient
                installed_version = pkg_resources.get_distribution(package).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                    outdated_packages.append((package, version, installed_version))

        # If packages are missing, check virtual environment first
        if missing_packages or outdated_packages:
            if not self._is_in_virtual_environment():
                self._prompt_create_environment()
        
        # Now install/upgrade packages
        for package, version in self.REQUIRED_PACKAGES.items():
            try:
                # Check if package is installed
                if package not in installed_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={version}"])
                    print(f"Successfully installed {package}")
                else:
                    # Check if version is sufficient
                    installed_version = pkg_resources.get_distribution(package).version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                        print(f"Upgrading {package} from {installed_version} to {version}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", f"{package}>={version}"])
                        print(f"Successfully upgraded {package}")
                    else:
                        print(f"[SUCCESS] {package} {installed_version} is already installed")
            except Exception as e:
                print(f"Error installing {package}: {str(e)}")
                raise

        print("\nAll required packages are installed and up to date!")
    
    def _is_in_virtual_environment(self):
        """Check if the script is running in a virtual environment"""
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    def _prompt_create_environment(self):
        """Prompt user to create a virtual environment"""
        print("\n" + "="*60)
        print("[WARNING] You are not in a virtual environment!")
        print("It's recommended to create a virtual environment before installing packages.")
        print("="*60)

        while True:
            response = input("\nWould you like to create a virtual environment? (y/n): ").lower().strip()
            
            if response in ['y', 'yes']:
                env_name = input(f"Enter environment name (default: env): ").strip()
                if not env_name:
                    env_name = "env"
                
                print(f"\nCreating virtual environment '{env_name}'...")
                try:
                    # Create virtual environment
                    subprocess.check_call([sys.executable, "-m", "venv", env_name])
                    print(f"[SUCCESS] Virtual environment '{env_name}' created successfully!")
                    
                    # Provide activation instructions
                    print(f"\nTo activate the environment, run:")
                    if os.name == 'nt':  # Windows
                        print(f"  {env_name}\\Scripts\\activate")
                    else:  # Unix/Linux/macOS
                        print(f"  source {env_name}/bin/activate")
                    
                    print(f"\nAfter activation, re-run this script to install packages.")
                    print("Exiting...")
                    sys.exit(0)
                    
                except Exception as e:
                    print(f"Error creating virtual environment: {str(e)}")
                    print("Please create the environment manually and try again.")
                    sys.exit(1)
                    
            elif response in ['n', 'no']:
                print("\nProceeding without virtual environment...")
                print("Note: This may install packages globally, which could cause conflicts.")
                break
            else:
                print("Please enter 'y' or 'n'.")
    def create_main_parser(self):
        """Create main argument parser"""
        parser = argparse.ArgumentParser(description='QIDIFEX')
        # Example selection
        parser.add_argument('--Model', type=str, 
                            choices = ['MC_triad'],
                            default='MC_triad',
                            help = 'Model to use')
        # Case selection
        if parser.parse_args().Model == 'MC_triad':
            parser.add_argument('--params_name', type=str, 
                                choices = ['cascade', 'equipart'],
                                default='cascade',
                                help='Case to use')
        # Seed
        parser.add_argument('--SEED', type=int, 
                            default=1234,
                            help='Seed for random number generator')
        # Hardware selection
        parser.add_argument('--DEVICE', type=str, 
                            choices = ['cpu','cuda:0','auto'],
                            default='cuda:0',
                            help='Device to use')
        # FEX controllersettings
        parser.add_argument('--CONTROLLER_LR', type=float, 
                            default=1e-1,
                            help='Learning rate for controller')
        parser.add_argument('--CONTROLLER_INPUT_SIZE', type=int, 
                            default=20,
                            help='Input size for controller')
        
        parser.add_argument('--CONTROLLER_HIDDEN_SIZE', type=int, 
                            default=30,
                            help='Hidden size for controller')

        parser.add_argument('--CONTROLLER_TOP_SAMPLES_FRACTION', type=float, 
                            default=0.25,
                            help='Top samples fraction for controller')
        
        parser.add_argument('--CONTROLLER_QUANTILE_METHOD', type=str, 
                            choices = ['linear', 'quantile'],
                            default='linear',
                            help='Quantile method for controller')
        # FEX training settings
        parser.add_argument('--EXPLORATION_ITERS', type=int, 
                            default=30,
                            help='Number of exploration iterations')
        parser.add_argument('--NUM_TREES', type=int, 
                            default=200,
                            help='Number of trees in inner iteration')
        
        # FEX training integration
        parser.add_argument('--INTEGRATOR_METHOD', type=str, 
                            choices = ['integration-based', 'derivative-based'],
                            default='integration-based',
                            help='Integration method')
        # FEX optimizer settings
        parser.add_argument('--FEX_LR', type=float, 
                            default=8e-3,
                            help='Learning rate for FEX optimizer')
        
        parser.add_argument('--TRAIN_EPOCHS_FIRST', type=int, 
                            default=20,
                            help='Number of epochs for first stage training')
        parser.add_argument('--TRAIN_EPOCHS_SECOND', type=int, 
                            default=2000,
                            help='Number of epochs for second stage training')
        # parser.add_argument('--TRAIN_GROUND_TRUTH',type=bool,default = True)
        # parser.add_argument('--MULTI_FEX_OPEN',type=float,default = True)
        #FEX-DM settings
        #parser.add_argument('--SECOND_STAGE_OPEN_BOOL',type=bool,default = False)
        parser.add_argument('--DIFF_SCALE',type=float,
                            default = 20,
                            help='Diffusion scale for FEX-DM')
        
        parser.add_argument('--ODESLOVER_TIME_STEPS',type=int,
                            default = 2000,
                            help='Number of time steps for ODE solver')
        
        parser.add_argument('--SHORT_SIZE',type=int,
                            default = 2048,
                            help='Short size for FEX-DM')
        
        parser.add_argument('--NN_SOLVER_LR',type=float,
                            default = 0.01,
                            help='Learning rate for NN solver')
        parser.add_argument('--NN_SOLVER_EPOCHS',type=int,
                            default = 2000,
                            help='Number of epochs for NN solver')
        
        parser.add_argument('--NUM_SAMPLES',type=int,
                            default = 10000,
                            help='Number of samples for DM training.')
        
        return parser
    
    def parse_args(self):
        """Parse command line arguments for main program"""
        main_parser = self.create_main_parser()
        return main_parser.parse_args()

    def load_triad_config_data(self, dimension, config_type='equipart'):
        """
        Lazy load numpy arrays for triad configuration
        
        Args:
            dimension (str): Dimension number ('1', '2', '3')
            config_type (str): Either 'equipart' or 'cascade'
            
        Returns:
            numpy.ndarray: Loaded array or None if loading fails
        """
        try:
            import numpy as np
        except ImportError:
            print("Warning: numpy is not installed. Cannot load triad configuration data.")
            return None
        
        if dimension not in self.TRIAD_MODEL_CONFIG:
            print(f"Warning: Dimension {dimension} not found in TRIAD_MODEL_CONFIG")
            return None
            
        if config_type not in ['equipart', 'cascade']:
            print(f"Warning: Config type {config_type} must be 'equipart' or 'cascade'")
            return None
            
        file_path = self.TRIAD_MODEL_CONFIG[dimension][f'op_seq_{config_type}_path']
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
            
        try:
            return np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {str(e)}")
            return None
    
    def get_triad_config(self, dimension, config_type='equipart'):
        """
        Get triad configuration with loaded data
        
        Args:
            dimension (str): Dimension number ('1', '2', '3')
            config_type (str): Either 'equipart' or 'cascade'
            
        Returns:
            dict: Configuration with loaded data
        """
        config = self.TRIAD_MODEL_CONFIG[dimension].copy()
        
        # Load the appropriate data
        data = self.load_triad_config_data(dimension, config_type)
        config[f'op_seq_{config_type}'] = data
        
        return config

# Create global instance
config = Config()

# Export commonly used attributes for backward compatibility
DIR_PROJECT = config.DIR_PROJECT
DIR_TRIAD = config.DIR_TRIAD
DIR_EQUIPART = config.DIR_EQUIPART
DIR_CASCADE = config.DIR_CASCADE

# Export functions
create_main_parser = config.create_main_parser
parse_args = config.parse_args
check_and_install_packages = config.check_and_install_packages
load_triad_config_data = config.load_triad_config_data
get_triad_config = config.get_triad_config


# Add project directory to sys.path
sys.path.append(DIR_PROJECT)

# =============================================================================
# CONVENIENT IMPORT HELPER
# =============================================================================

def get_config():
    """Get the global config instance - loads only once"""
    return config



# Make it easy to import specific items
__all__ = [
    'config',
    'get_config',
    'create_main_parser',
    'parse_args',
    'check_and_install_packages',
    'load_triad_config_data',
    'get_triad_config',
    'DIR_PROJECT',
    'DIR_TRIAD',
    'DIR_EQUIPART',
    'DIR_CASCADE',
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Configuration module loaded successfully!")

    # Uncomment to check packages
    config.check_and_install_packages()