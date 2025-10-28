import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def configure():
    warnings.filterwarnings("ignore")
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
