from aegnn.datasets.base.event_dm import EventDataModule

from aegnn.datasets.ncaltech101 import NCaltech101
from aegnn.datasets.ncars import NCars
from aegnn.datasets.gen1 import Gen1
from aegnn.datasets.syn import Syn
from aegnn.datasets.syn_new import Syn_New
from aegnn.datasets.syn_related import Syn_Related
from aegnn.datasets.syn_heatmap import Syn_Heatmap


################################################################################################
# Access functions #############################################################################
################################################################################################
def by_name(name: str) -> EventDataModule.__class__:
    if name.lower() == "ncaltech101":
        return NCaltech101
    elif name.lower() == "ncars":
        return NCars
    elif name.lower() == "gen1":
        return Gen1
    elif name.lower() == "syn":
        return Syn
    elif name.lower() == "syn_new":
        return Syn_New
    elif name.lower() == "syn_related":
        return Syn_Related
    elif name.lower() == "syn_heatmap":
        return Syn_Heatmap
    else:
        raise NotImplementedError(f"Dataset with name {name} is not known!")
